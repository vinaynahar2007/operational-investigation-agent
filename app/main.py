"""Command-line entry point for the deterministic anomaly scan and investigation.

Usage
-----
    python -m app.main
    python -m app.main --scan
    python -m app.main --json
    python -m app.main --machine M04
    python -m app.main --investigate M04
    python -m app.main --investigate M04 --json

No LLM is involved. This runs only deterministic detection and investigation
over the dataset.
"""

from __future__ import annotations

import argparse
import json
import sys

from app.agents.investigator import investigate
from app.tools.anomalies import detect_anomalies, findings_by_machine
from app.tools.dataset import validate_dataset

RULE = "=" * 40
THIN = "-" * 40


def format_magnitude(value: float) -> str:
    return "{0:+.1f}%".format(value)


def _yes_no(value) -> str:
    return "yes" if value else "no"


def _trace_entry(result: dict, step: str):
    """Find a trace entry by stage name, never by position."""
    for entry in result.get("trace", []):
        if entry["step"] == step:
            return entry
    return None


def _reasoning_status(result: dict) -> str:
    entry = _trace_entry(result, "LLM_REASONING")
    return entry["status"].upper() if entry else "SKIPPED"


def _reasoning_field(result: dict, field: str) -> str:
    entry = _trace_entry(result, "LLM_REASONING")
    return entry.get(field, "unknown") if entry else "unknown"


def render(findings: list) -> str:
    """Human-readable scan report."""
    lines = [RULE, "OPERATIONAL ANOMALY SCAN", RULE, ""]

    if not findings:
        lines.append("No anomalies detected.")
        lines.append("")
        return "\n".join(lines)

    grouped = findings_by_machine(findings)
    for machine_id in sorted(grouped):
        for finding in grouped[machine_id]:
            lines.append(machine_id)
            lines.append(
                "{0} -> {1}".format(finding["start_date"], finding["end_date"])
            )
            lines.append("metric: {0}".format(finding["metric"]))
            lines.append("magnitude: {0}".format(format_magnitude(finding["magnitude_pct"])))
            lines.append("severity: {0}".format(finding["severity"]))
            lines.append("confidence: {0:.2f}".format(finding["confidence"]))
            lines.append("type: {0}".format(finding["anomaly_type"]))
            lines.append("days: {0}".format(finding["days"]))
            lines.append("")

    lines.append(THIN)
    lines.append("{0} machine(s), {1} anomaly window(s)".format(len(grouped), len(findings)))
    lines.append(THIN)
    return "\n".join(lines)


def render_investigation(result: dict) -> str:
    """Human-readable investigation report.

    Sections are labelled OBSERVED (measured) or INFERRED (reasoned) so a
    reader can always tell which is which.
    """
    lines = [RULE, "OPERATIONAL INVESTIGATION", RULE, ""]

    lines.append("QUESTION")
    lines.append(result["question"])
    lines.append("")

    lines.append("STATUS: {0}".format(result["status"]))
    lines.append("")

    lines.append("ANOMALY")
    anomaly = result["anomaly"]
    if not anomaly:
        lines.append("  none detected")
    else:
        lines.append("  {0} -> {1}".format(anomaly["start_date"], anomaly["end_date"]))
        lines.append("  metric: {0}".format(anomaly["metric"]))
        lines.append(
            "  magnitude: {0}".format(format_magnitude(anomaly["magnitude_pct"]))
        )
        lines.append("  severity: {0}".format(anomaly["severity"]))
    lines.append("")

    lines.append("EVIDENCE (OBSERVED)")
    evidence = result["evidence"]
    if not evidence:
        lines.append("  not collected")
    else:
        metrics = evidence["metrics"]
        baseline_days = evidence["baseline_window"]["days"]
        lines.append(
            "  window {0} -> {1} vs baseline {2} -> {3}".format(
                evidence["window"]["start_date"],
                evidence["window"]["end_date"],
                evidence["baseline_window"]["start_date"],
                evidence["baseline_window"]["end_date"],
            )
        )
        for label, key in (
            ("energy", "energy_kwh"),
            ("runtime", "runtime_h"),
            ("intensity", "intensity"),
            ("efficiency_factor", "efficiency_factor"),
        ):
            lines.append(
                "  {0:<17} {1} (ratio {2})".format(
                    label,
                    format_magnitude(metrics[key]["change_pct"]),
                    metrics[key]["ratio"],
                )
            )
        lines.append(
            "  {0:<17} {1} (ratio {2})".format(
                "production",
                format_magnitude(evidence["production"]["change_pct"]),
                evidence["production"]["ratio"],
            )
        )
        lines.append(
            "  maintenance records in window: {0}".format(
                len(evidence["maintenance_records"])
            )
        )
        lines.append(
            "  technician notes in window: {0}".format(
                len(evidence["technician_notes"])
            )
        )
        lines.append(
            "  prior incidents for machine: {0}".format(
                len(evidence["historical_incidents"])
            )
        )
        for note in evidence["technician_notes"]:
            lines.append("    note ({0}): {1}".format(note["date"], note["note"]))
    lines.append("")

    lines.append("HYPOTHESES (INFERRED)")
    for hypothesis in result["hypotheses"]:
        lines.append(
            "  {0}: {1} (confidence {2:.2f})".format(
                hypothesis["name"], hypothesis["status"], hypothesis["confidence"]
            )
        )
        lines.append("    reason: {0}".format(hypothesis["reason"]))
        for item in hypothesis["evidence"]:
            lines.append("    - {0}".format(item))
    lines.append("")

    lines.append("PROBABLE ROOT CAUSE (INFERRED)")
    root_cause = result["root_cause"]
    if not root_cause:
        lines.append("  inconclusive - no hypothesis was supported.")
        lines.append("  No root cause is asserted.")
    else:
        lines.append("  category: {0}".format(root_cause["category"]))
        lines.append("  statement: {0}".format(root_cause["statement"]))
        lines.append("  confidence: {0:.2f}".format(root_cause["confidence"]))
        for item in root_cause["supporting_evidence"]:
            lines.append("    - {0}".format(item))
    lines.append("")

    lines.append("MEMORY SEARCH (ADVISORY)")
    matches = result.get("memory_matches") or []
    if not matches:
        lines.append(THIN)
        lines.append("  No similar historical investigation found.")
        lines.append(THIN)
    else:
        lines.append(THIN)
        for match in matches:
            case = match["case"]
            lines.append(
                "  Case #{0} (similarity score {1}, field-based)".format(
                    match["id"], match["score"]
                )
            )
            lines.append(
                "    Machine: {0}".format(case["machine_id"])
            )
            lines.append(
                "    Root cause: {0}".format(case["root_cause_category"])
            )
            lines.append(
                "    Intervention: {0}".format(case["intervention_description"])
            )
            lines.append(
                "    Verification: {0} (simulation: {1})".format(
                    case["verification_status"],
                    "yes" if case["simulation"] else "no",
                )
            )
            lines.append(
                "    Why considered similar: {0}".format("; ".join(match["reason"]))
            )
        lines.append(
            "  Historical precedent only. Current evidence remains authoritative."
        )
        lines.append(THIN)
    lines.append("")

    lines.append("INTERVENTION (SIMULATED)")
    intervention = result["intervention"]
    if not intervention:
        lines.append("  not simulated")
    else:
        lines.append(THIN)
        lines.append("  Type: {0}".format(intervention["type"]))
        lines.append("  Description: {0}".format(intervention["description"]))
        lines.append(
            "  Baseline energy: {0:.2f} kWh/day".format(
                intervention["baseline_energy"]
            )
        )
        lines.append(
            "  Projected energy: {0:.2f} kWh/day".format(
                intervention["projected_energy"]
            )
        )
        lines.append(
            "  Projected reduction: {0:.2f}%".format(
                intervention["projected_reduction_percent"]
            )
        )
        lines.append("  SIMULATED: {0}".format(_yes_no(intervention["simulation"])))
        lines.append(
            "  No physical change was made; this is an arithmetic projection."
        )
        lines.append(THIN)
    lines.append("")

    lines.append("VERIFICATION (SIMULATED)")
    verification = result["verification"]
    if not verification:
        lines.append("  not verified")
    else:
        lines.append(THIN)
        lines.append("  Status: {0}".format(verification["status"]))
        lines.append(
            "  Observed reduction: {0:.2f}%".format(
                verification["observed_reduction_percent"]
            )
        )
        lines.append(
            "  Threshold: {0:.2f}%".format(verification["threshold_percent"])
        )
        lines.append("  SIMULATED: {0}".format(_yes_no(verification["simulation"])))
        lines.append("  Reason: {0}".format(verification["reason"]))
        lines.append(
            "  This verifies the arithmetic projection only; no machine was "
            "measured."
        )
        lines.append(THIN)
    lines.append("")

    lines.append("LLM REASONING")
    reasoning = result.get("llm_reasoning")
    if not reasoning:
        lines.append("  SKIPPED")
        lines.append("  Reason: LLM unavailable")
        lines.append("  Fallback: deterministic investigator")
    elif not reasoning.get("available"):
        lines.append(THIN)
        lines.append("  Status: {0}".format(_reasoning_status(result)))
        lines.append("  Reason: {0}".format(reasoning.get("reason", "")))
        lines.append("  Fallback: deterministic investigator")
        lines.append(THIN)
    else:
        lines.append(THIN)
        lines.append("  Provider: {0}".format(_reasoning_field(result, "provider")))
        lines.append("  Model: {0}".format(_reasoning_field(result, "model")))
        lines.append("  Status: {0}".format(_reasoning_status(result)))
        lines.append("  Authoritative source: {0}".format(
            reasoning.get("authoritative_source", "deterministic")
        ))
        lines.append("  Reasoning: {0}".format(reasoning.get("reasoning_summary", "")))
        if reasoning.get("root_cause_explanation"):
            lines.append(
                "  Root cause explanation: {0}".format(
                    reasoning["root_cause_explanation"]
                )
            )
        if reasoning.get("recommended_intervention_explanation"):
            lines.append(
                "  Intervention explanation (SIMULATED): {0}".format(
                    reasoning["recommended_intervention_explanation"]
                )
            )
        for conflict in reasoning.get("conflicts", []):
            lines.append(
                "  CONFLICT {0}: model said {1}, evidence says {2} "
                "(kept deterministic)".format(
                    conflict["hypothesis"],
                    conflict["model_status"],
                    conflict["deterministic_status"],
                )
            )
        lines.append(
            "  Advisory only: all numeric values come from deterministic tools."
        )
        lines.append(THIN)
    lines.append("")

    lines.append("MEMORY STORE")
    lines.append(THIN)
    record_id = result.get("memory_record_id")
    if record_id is None:
        lines.append("  Stored investigation ID: none")
        entry = _trace_entry(result, "MEMORY_STORE")
        if entry:
            lines.append("  Reason: {0}".format(entry["summary"]))
    else:
        lines.append("  Stored investigation ID: {0}".format(record_id))
        lines.append("  Simulation: yes")
        lines.append(
            "  This is a simulated outcome, not a measured real-world result."
        )
    lines.append(THIN)
    lines.append("")

    lines.append("RECOMMENDED NEXT ACTION")
    lines.append("  {0}".format(result["recommended_next_action"] or "none"))
    lines.append("")

    lines.append("INVESTIGATION TRACE")
    for entry in result["trace"]:
        lines.append(
            "  [{0}] {1}: {2}".format(
                entry["status"].upper(), entry["step"], entry["summary"]
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.main",
        description="Deterministic anomaly scan and investigation.",
    )
    parser.add_argument(
        "--scan", action="store_true", help="Run the scan (default behaviour)."
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit findings as JSON."
    )
    parser.add_argument(
        "--machine", help="Restrict output to a single machine id."
    )
    parser.add_argument(
        "--investigate",
        metavar="MACHINE_ID",
        help="Run a full investigation for one machine.",
    )
    parser.add_argument(
        "--question", help="Question text for --investigate."
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Do not print the dataset validation result first.",
    )
    args = parser.parse_args(argv)

    if not args.skip_validation:
        report = validate_dataset()
        status = "OK" if report["ok"] else "FAILED"
        print("Dataset validation: {0}".format(status))
        for error in report["errors"]:
            print("  - {0}".format(error))
        print("")

    if args.investigate:
        result = investigate(args.investigate, args.question)
        if args.json:
            print(json.dumps(result, indent=2, default=str))
            return 0
        print(render_investigation(result))
        return 0

    findings = detect_anomalies()
    if args.machine:
        findings = [f for f in findings if f["machine_id"] == args.machine]

    if args.json:
        print(json.dumps(findings, indent=2))
        return 0

    print(render(findings))
    return 0


if __name__ == "__main__":
    sys.exit(main())