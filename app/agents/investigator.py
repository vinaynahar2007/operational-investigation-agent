"""Deterministic investigation state machine.

Flow
----
    DETECT -> EVIDENCE -> HYPOTHESES -> EVALUATE -> ROOT_CAUSE
           -> INTERVENTION -> VERIFY -> COMPLETE

Each step consumes the previous step's output and appends to a trace, so the
reasoning path is auditable rather than hidden behind a single answer.

Strict separation of observed, inferred and simulated
---------------------------------------------------
``app.tools.evidence`` produces OBSERVED values (measured ratios, records).
This module produces INFERRED statements (hypothesis verdicts, root cause).
``app.tools.intervention`` and ``app.tools.verification`` produce SIMULATED
values. Nothing here changes a measurement, and no physical machine is ever
touched: the intervention is arithmetic and the verification only checks that
arithmetic against a threshold.

No LLM is involved. Every verdict is a comparison of two measured numbers
against a threshold declared in this file.
"""

from __future__ import annotations

from app.tools.anomalies import detect_anomalies_for_machine
from app.llm.adapter import (
    LLMError,
    build_evidence_contract,
    get_adapter,
    reconcile_reasoning,
    unavailable_reasoning,
)
from app.tools.evidence import gather_evidence, machine_info
from app.tools.intervention import (
    expected_reduction_for,
    intervention_for,
    simulate_intervention,
)
from app.tools.memory import search_memory_for_result, store_investigation
from app.tools.verification import DEFAULT_THRESHOLD_PCT, verify_intervention

# --- investigation stages -------------------------------------------------

STAGE_DETECT = "DETECT"
STAGE_EVIDENCE = "EVIDENCE"
STAGE_HYPOTHESES = "HYPOTHESES"
STAGE_LLM_REASONING = "LLM_REASONING"
STAGE_EVALUATE = "EVALUATE"
STAGE_ROOT_CAUSE = "ROOT_CAUSE"
STAGE_MEMORY_SEARCH = "MEMORY_SEARCH"
STAGE_INTERVENTION = "INTERVENTION"
STAGE_VERIFY = "VERIFY"
STAGE_MEMORY_STORE = "MEMORY_STORE"
STAGE_COMPLETE = "COMPLETE"

TRACE_STAGES = (
    STAGE_DETECT,
    STAGE_EVIDENCE,
    STAGE_HYPOTHESES,
    STAGE_LLM_REASONING,
    STAGE_EVALUATE,
    STAGE_ROOT_CAUSE,
    STAGE_MEMORY_SEARCH,
    STAGE_INTERVENTION,
    STAGE_VERIFY,
    STAGE_MEMORY_STORE,
    STAGE_COMPLETE,
)

# --- hypothesis names -----------------------------------------------------

HYPOTHESIS_PRODUCTION = "production_increase"
HYPOTHESIS_RUNTIME = "runtime_scheduling_increase"
HYPOTHESIS_EQUIPMENT = "equipment_efficiency_degradation"
HYPOTHESIS_MAINTENANCE = "maintenance_operational_condition"
HYPOTHESIS_HISTORICAL = "historical_similar_incident"

CATEGORY_PRODUCTION = "production_demand"
CATEGORY_RUNTIME = "operational_scheduling"
CATEGORY_EQUIPMENT = "equipment_efficiency"
CATEGORY_MAINTENANCE = "maintenance_condition"
CATEGORY_HISTORICAL = "historical_precedent"

STATUS_SUPPORTED = "supported"
STATUS_REJECTED = "rejected"
STATUS_INCONCLUSIVE = "inconclusive"

# --- decision thresholds --------------------------------------------------

#: A metric is "materially elevated" above this ratio against its own baseline.
MATERIAL_RISE = 1.08

#: A metric is "flat" below this ratio, i.e. inside normal variation.
FLAT = 1.05

#: Production must explain at least this share of the energy rise to count as
#: the explanation. Anything smaller is a real but insufficient contribution.
PRODUCTION_EXPLANATION_SHARE = 0.5

#: Confidence assigned to a verdict is built from how decisively the measured
#: ratio sits outside the flat band, never from a hand-picked per-case value.
CONFIDENCE_FLOOR = 0.5
CONFIDENCE_CEILING = 0.95


class Investigation:
    """Runs the deterministic stages and accumulates a trace."""

    def __init__(
        self,
        machine_id: str,
        question: str | None = None,
        adapter=None,
        memory_db=None,
    ):
        self.machine_id = machine_id
        self.question = question or "Why did {0} behave abnormally?".format(machine_id)
        # Optional reasoning layer. None or unavailable means deterministic only.
        self.adapter = adapter
        # None uses the configured database; False disables memory entirely.
        self.memory_db = memory_db
        self.llm_reasoning = None
        self.memory_matches: list = []
        self.memory_record_id = None
        self.trace: list = []
        self.anomaly = None
        self.evidence = None
        self.hypotheses: list = []
        self.root_cause = None
        self.intervention = None
        self.verification = None
        self.recommended_next_action = None
        self.status = "pending"

    def _record(
        self,
        step: str,
        status: str,
        summary: str,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        entry = {"step": step, "status": status, "summary": summary}
        if provider is not None:
            entry["provider"] = provider
        if model is not None:
            entry["model"] = model
        self.trace.append(entry)

    def _halt(self, step: str, summary: str, status: str = "failed") -> None:
        """Stop the run, marking every later stage as skipped.

        ``status`` defaults to ``failed`` but callers pass their own terminal
        state, because not finding an anomaly is a normal outcome rather than
        an error.
        """
        self.status = status
        self._record(step, "failed" if status == "failed" else "skipped", summary)
        start = TRACE_STAGES.index(step) + 1
        for remaining in TRACE_STAGES[start:]:
            self._record(
                remaining,
                "skipped",
                "Not reached: investigation halted at {0}.".format(step),
            )

    def run(self) -> dict:
        anomaly = self._detect()
        if anomaly is None:
            return self.result()

        evidence = self._collect_evidence(anomaly)
        if evidence is None:
            return self.result()

        self.hypotheses = build_hypotheses(evidence)
        self._record(
            STAGE_HYPOTHESES,
            "complete",
            "Raised {0} candidate hypotheses: {1}.".format(
                len(self.hypotheses),
                ", ".join(h["name"] for h in self.hypotheses),
            ),
        )

        evaluate_hypotheses(self.hypotheses, evidence)
        self._run_llm_reasoning()

        self._record(
            STAGE_EVALUATE,
            "complete",
            "Evaluated against measured ratios: {0}.".format(
                _verdict_summary(self.hypotheses)
            ),
        )

        self.root_cause = select_root_cause(self.hypotheses, evidence)
        if self.root_cause is None:
            self.status = "inconclusive"
            self._record(
                STAGE_ROOT_CAUSE,
                "inconclusive",
                "No hypothesis reached supported status; no root cause asserted.",
            )
            self.recommended_next_action = _next_action_when_inconclusive(evidence)
            # No cause means no defensible intervention; do not simulate one.
            self._record(
                STAGE_INTERVENTION,
                "skipped",
                "Skipped because no probable root cause was established.",
            )
            self._record(
                STAGE_VERIFY,
                "skipped",
                "Skipped because no intervention was simulated.",
            )
            self._record(
                STAGE_MEMORY_STORE,
                "skipped",
                "Skipped because the investigation was inconclusive.",
            )
        else:
            self._record(
                STAGE_ROOT_CAUSE,
                "complete",
                "Probable cause: {0}".format(self.root_cause["statement"]),
            )
            self.recommended_next_action = self.root_cause["recommended_next_action"]
            # Memory is consulted before acting, but never overrides evidence.
            self._search_memory()
            if not self._simulate_intervention():
                return self.result()
            if not self._verify_projection():
                return self.result()
            # Promote the terminal status before storing, so the stored record
            # reflects a genuinely completed, verified investigation.
            if self.status in ("pending", "complete"):
                self.status = "complete"
            self._store_in_memory()

        # A previous stage may already have set a more specific terminal status
        # (inconclusive or failed); never overwrite it with a clean completion.
        if self.status in ("pending", "complete"):
            self.status = "complete"
        self._record(
            STAGE_COMPLETE,
            "complete",
            "Investigation finished with status '{0}'.".format(self.status),
        )
        return self.result()

    def _simulate_intervention(self) -> bool:
        """SIMULATED projection for the selected root cause.

        Returns False (and records a failed stage) when the projection cannot
        be built, so the investigation is never reported as complete on the
        back of a missing simulation.
        """
        category = self.root_cause["category"]
        baseline = self._baseline_energy()
        if baseline is None or baseline <= 0:
            self._record(
                STAGE_INTERVENTION,
                "failed",
                "No usable baseline energy available; cannot simulate.",
            )
            self.status = "failed"
            self._record(
                STAGE_VERIFY, "skipped", "Skipped because no simulation was created."
            )
            self._record(
                STAGE_COMPLETE, "skipped", "Not reached: simulation failed."
            )
            return False

        excess = self.evidence["metrics"]["energy_kwh"]["change_pct"]
        reduction = expected_reduction_for(category, excess)

        try:
            projection = simulate_intervention(
                baseline, intervention_for(category), reduction
            )
        except ValueError as error:
            self._record(STAGE_INTERVENTION, "failed", "Simulation rejected: {0}".format(error))
            self.status = "failed"
            self._record(
                STAGE_VERIFY, "skipped", "Skipped because no simulation was created."
            )
            self._record(
                STAGE_COMPLETE, "skipped", "Not reached: simulation failed."
            )
            return False

        self.intervention = {
            "type": category,
            "description": projection["intervention"],
            "baseline_energy": projection["baseline_energy"],
            "projected_energy": projection["projected_energy"],
            "projected_reduction_percent": projection["projected_reduction_percent"],
            "simulation": True,
        }
        self._record(
            STAGE_INTERVENTION,
            "complete",
            "SIMULATED {0} Baseline {1:.2f} kWh/day -> projected {2:.2f} "
            "kWh/day, an assumed {3:.2f}% reduction. No physical change was "
            "made.".format(
                projection["intervention"],
                projection["baseline_energy"],
                projection["projected_energy"],
                projection["projected_reduction_percent"],
            ),
        )
        return True

    def _verify_projection(self) -> bool:
        """Check the simulated projection against the verification threshold."""
        try:
            outcome = verify_intervention(
                self.intervention["baseline_energy"],
                self.intervention["projected_energy"],
                self.intervention["projected_reduction_percent"],
                DEFAULT_THRESHOLD_PCT,
            )
        except ValueError as error:
            self._record(STAGE_VERIFY, "failed", "Verification rejected: {0}".format(error))
            self.status = "failed"
            self._record(
                STAGE_COMPLETE, "skipped", "Not reached: verification failed."
            )
            return False

        self.verification = outcome
        passed = outcome["status"] == "PASS"
        self._record(
            STAGE_VERIFY,
            "complete" if passed else "failed",
            "{0}".format(outcome["reason"]),
        )
        if not passed:
            # A failed projection must not be reported as a clean completion.
            self.status = "verification_failed"
        return True

    def _baseline_energy(self):
        """OBSERVED mean daily energy over the anomaly window."""
        return self.evidence["metrics"]["energy_kwh"]["window_value"]

    def _search_memory(self) -> None:
        """Look for previously investigated similar cases.

        Memory is advisory. Nothing found here can change the current root
        cause, hypothesis statuses or measurements; it can only add precedent
        that the reasoning layer may mention.
        """
        if self.memory_db is False:
            self.memory_matches = []
            self._record(
                STAGE_MEMORY_SEARCH,
                "skipped",
                "Memory lookup disabled for this run.",
            )
            return

        try:
            matches = search_memory_for_result(self.result(), db_path=self.memory_db)
        except Exception as error:  # memory must never break an investigation
            self.memory_matches = []
            self._record(
                STAGE_MEMORY_SEARCH,
                "failed",
                "Memory lookup failed ({0}); continuing without it.".format(error),
            )
            return

        self.memory_matches = matches
        if not matches:
            self._record(
                STAGE_MEMORY_SEARCH,
                "complete",
                "No similar historical investigation found.",
            )
            return

        top = matches[0]
        self._record(
            STAGE_MEMORY_SEARCH,
            "complete",
            "Found {0} similar case(s). Closest: #{1} (score {2}) - {3}. "
            "Historical precedent only; current evidence remains "
            "authoritative.".format(
                len(matches),
                top["id"],
                top["score"],
                "; ".join(top["reason"]),
            ),
        )

    def _store_in_memory(self) -> None:
        """Store this investigation as operational memory after verification."""
        if self.memory_db is False:
            self.memory_record_id = None
            self._record(
                STAGE_MEMORY_STORE,
                "skipped",
                "Memory storage disabled for this run.",
            )
            return

        if self.status != "complete":
            self.memory_record_id = None
            self._record(
                STAGE_MEMORY_STORE,
                "skipped",
                "Not stored because the investigation did not complete "
                "(status '{0}').".format(self.status),
            )
            return

        try:
            record_id = store_investigation(self.result(), db_path=self.memory_db)
        except Exception as error:  # storage failure must not fail the run
            self.memory_record_id = None
            self._record(
                STAGE_MEMORY_STORE,
                "failed",
                "Memory storage failed ({0}); investigation still "
                "complete.".format(error),
            )
            return

        if record_id is None:
            self.memory_record_id = None
            self._record(
                STAGE_MEMORY_STORE,
                "skipped",
                "Not stored because the outcome was not a verified success.",
            )
            return

        self.memory_record_id = record_id
        self._record(
            STAGE_MEMORY_STORE,
            "complete",
            "Stored as investigation #{0} (simulation: yes; a simulated "
            "outcome, not a measured result).".format(record_id),
        )

    def _run_llm_reasoning(self) -> None:
        """Optional LLM reasoning over the evidence.

        This stage is advisory. It never changes a measurement, never changes a
        hypothesis status, and never changes the selected root cause. Any
        failure degrades to deterministic reasoning rather than failing the
        investigation.
        """
        adapter = self.adapter
        if adapter is None:
            adapter = get_adapter()

        if not adapter.is_available():
            self.llm_reasoning = unavailable_reasoning(
                "LLM unavailable; deterministic reasoning used."
            )
            self._record(
                STAGE_LLM_REASONING,
                "skipped",
                "LLM unavailable; deterministic reasoning used.",
                provider=self._provider_label(adapter),
                model=self._model_label(adapter),
            )
            return

        contract = build_evidence_contract(self.evidence)

        try:
            reasoning = adapter.reason(contract)
        except LLMError as error:
            # A model failure must never fail the investigation.
            self.llm_reasoning = unavailable_reasoning(
                "LLM reasoning failed ({0}); deterministic reasoning used.".format(
                    error
                )
            )
            self._record(
                STAGE_LLM_REASONING,
                "failed",
                "LLM reasoning failed ({0}); deterministic reasoning used.".format(
                    error
                ),
                provider=self._provider_label(adapter),
                model=self._model_label(adapter),
            )
            return
        except Exception as error:  # defensive: never break the pipeline
            self.llm_reasoning = unavailable_reasoning(
                "LLM reasoning error ({0}); deterministic reasoning used.".format(
                    error
                )
            )
            self._record(
                STAGE_LLM_REASONING,
                "failed",
                "LLM reasoning error ({0}); deterministic reasoning used.".format(
                    error
                ),
                provider=self._provider_label(adapter),
                model=self._model_label(adapter),
            )
            return

        self.llm_reasoning = reconcile_reasoning(
            self.hypotheses, reasoning, contract
        )
        conflict_note = ""
        if self.llm_reasoning["conflicts"]:
            conflict_note = " {0} interpretation conflict(s) recorded; the "\
                "deterministic verdict was kept.".format(
                    len(self.llm_reasoning["conflicts"])
                )
        self._record(
            STAGE_LLM_REASONING,
            "complete",
            "Model reasoning received from {0}/{1}. Advisory only.{2}".format(
                self._provider_label(adapter),
                self._model_label(adapter),
                conflict_note,
            ),
            provider=self._provider_label(adapter),
            model=self._model_label(adapter),
        )

    def _provider_label(self, adapter) -> str:
        try:
            return adapter.describe().get("provider", "unknown")
        except Exception:
            return "unknown"

    def _model_label(self, adapter) -> str:
        try:
            return adapter.describe().get("model", "unknown")
        except Exception:
            return "unknown"

    def _detect(self):
        # An unknown machine is a different situation from a clean machine:
        # report it as such rather than implying we looked and found nothing.
        if not machine_info(self.machine_id)["found"]:
            self._record(
                STAGE_DETECT,
                "failed",
                "Machine {0} is not present in machines.csv.".format(self.machine_id),
            )
            self._halt(
                STAGE_EVIDENCE,
                "Cannot gather evidence for unknown machine {0}.".format(
                    self.machine_id
                ),
                status="failed",
            )
            return None

        findings = detect_anomalies_for_machine(self.machine_id)
        if not findings:
            self._record(
                STAGE_DETECT,
                "complete",
                "No anomaly detected for {0}; nothing to investigate.".format(
                    self.machine_id
                ),
            )
            self._halt(
                STAGE_EVIDENCE,
                "Skipped because no anomaly window was found for {0}.".format(
                    self.machine_id
                ),
                status="no_anomaly",
            )
            return None

        # The most severe window drives the investigation; others are retained.
        self.anomaly = findings[0]
        self.anomaly["additional_windows"] = findings[1:]
        self._record(
            STAGE_DETECT,
            "complete",
            "Detected {0} anomaly window(s); investigating {1} -> {2} "
            "({3}, {4:+.1f}%).".format(
                len(findings),
                self.anomaly["start_date"],
                self.anomaly["end_date"],
                self.anomaly["metric"],
                self.anomaly["magnitude_pct"],
            ),
        )
        return self.anomaly

    def _collect_evidence(self, anomaly):
        evidence = gather_evidence(self.machine_id, anomaly)
        self.evidence = evidence

        if not evidence["data_quality"]["machine_known"]:
            self._halt(
                STAGE_EVIDENCE,
                "Machine {0} is not present in machines.csv.".format(self.machine_id),
            )
            return None

        metrics = evidence["metrics"]
        self._record(
            STAGE_EVIDENCE,
            "complete",
            "Measured against the preceding {0}-day baseline: energy {1}, "
            "runtime {2}, intensity {3}, efficiency {4}, production {5}.".format(
                evidence["baseline_window"]["days"],
                _ratio_text(metrics["energy_kwh"]),
                _ratio_text(metrics["runtime_h"]),
                _ratio_text(metrics["intensity"]),
                _ratio_text(metrics["efficiency_factor"]),
                _ratio_text(evidence["production"]),
            ),
        )

        if not evidence["data_quality"]["sufficient"]:
            self._halt(
                STAGE_EVIDENCE,
                "Only {0}/{1} required metrics available; too thin to support "
                "a conclusion.".format(
                    evidence["data_quality"]["metrics_available"],
                    evidence["data_quality"]["metrics_expected"],
                ),
            )
            return None
        return evidence

    def result(self) -> dict:
        return {
            "machine_id": self.machine_id,
            "question": self.question,
            "status": self.status,
            "anomaly": self.anomaly,
            "evidence": self.evidence,
            "hypotheses": self.hypotheses,
            "root_cause": self.root_cause,
            "intervention": self.intervention,
            "verification": self.verification,
            "llm_reasoning": self.llm_reasoning,
            "memory_matches": self.memory_matches,
            "memory_record_id": self.memory_record_id,
            "recommended_next_action": self.recommended_next_action,
            "trace": self.trace,
        }


def investigate(
    machine_id: str,
    question: str | None = None,
    adapter=None,
    memory_db=None,
) -> dict:
    """Run the full deterministic investigation for one machine.

    ``adapter`` and ``memory_db`` are optional. Pass ``memory_db=False`` to run
    without operational memory; pass a path to isolate the database.
    """
    return Investigation(machine_id, question, adapter, memory_db).run()


def build_hypotheses(evidence: dict) -> list:
    """Every candidate hypothesis, each starting as ``inconclusive``.

    Candidates are always the same fixed set; relevance (whether a record type
    exists at all) is handled during evaluation, so the list stays comparable
    between investigations.
    """
    return [
        {
            "name": HYPOTHESIS_PRODUCTION,
            "category": CATEGORY_PRODUCTION,
            "status": STATUS_INCONCLUSIVE,
            "evidence": [],
            "reason": "Not yet evaluated.",
            "confidence": 0.0,
        },
        {
            "name": HYPOTHESIS_RUNTIME,
            "category": CATEGORY_RUNTIME,
            "status": STATUS_INCONCLUSIVE,
            "evidence": [],
            "reason": "Not yet evaluated.",
            "confidence": 0.0,
        },
        {
            "name": HYPOTHESIS_EQUIPMENT,
            "category": CATEGORY_EQUIPMENT,
            "status": STATUS_INCONCLUSIVE,
            "evidence": [],
            "reason": "Not yet evaluated.",
            "confidence": 0.0,
        },
        {
            "name": HYPOTHESIS_MAINTENANCE,
            "category": CATEGORY_MAINTENANCE,
            "status": STATUS_INCONCLUSIVE,
            "evidence": [],
            "reason": "Not yet evaluated.",
            "confidence": 0.0,
        },
        {
            "name": HYPOTHESIS_HISTORICAL,
            "category": CATEGORY_HISTORICAL,
            "status": STATUS_INCONCLUSIVE,
            "evidence": [],
            "reason": "Not yet evaluated.",
            "confidence": 0.0,
        },
    ]


def evaluate_hypotheses(hypotheses: list, evidence: dict) -> list:
    """Score each hypothesis against the measured ratios, in place.

    Rules are purely comparative and reference no machine id or date:

    - production is supported only if the production rise explains a majority
      of the energy rise;
    - runtime and equipment are mutually exclusive explanations, resolved by
      which of the two actually moved while the other stayed flat;
    - maintenance and historical hypotheses rest on the presence of matching
      records, never on the size of the anomaly.
    """
    metrics = evidence["metrics"]
    energy_ratio = metrics["energy_kwh"]["ratio"]
    runtime_ratio = metrics["runtime_h"]["ratio"]
    intensity_ratio = metrics["intensity"]["ratio"]
    efficiency_ratio = metrics["efficiency_factor"]["ratio"]
    production_ratio = evidence["production"]["ratio"]

    for hypothesis in hypotheses:
        name = hypothesis["name"]

        if name == HYPOTHESIS_PRODUCTION:
            _judge_production(hypothesis, energy_ratio, production_ratio)
        elif name == HYPOTHESIS_RUNTIME:
            _judge_runtime(hypothesis, runtime_ratio, intensity_ratio)
        elif name == HYPOTHESIS_EQUIPMENT:
            _judge_equipment(hypothesis, intensity_ratio, efficiency_ratio, runtime_ratio)
        elif name == HYPOTHESIS_MAINTENANCE:
            _judge_maintenance(hypothesis, evidence)
        elif name == HYPOTHESIS_HISTORICAL:
            _judge_historical(hypothesis, evidence)

    return hypotheses


def _judge_production(hypothesis, energy_ratio, production_ratio):
    if energy_ratio is None or production_ratio is None:
        _set(
            hypothesis,
            STATUS_INCONCLUSIVE,
            [],
            "Production or energy comparison unavailable.",
            0.0,
        )
        return

    evidence = [
        "OBSERVED energy ratio {0}".format(_fmt_ratio(energy_ratio)),
        "OBSERVED production ratio {0}".format(_fmt_ratio(production_ratio)),
    ]

    if production_ratio < 1.0:
        _set(
            hypothesis,
            STATUS_REJECTED,
            evidence,
            "Production did not rise (ratio {0}); a demand increase cannot "
            "explain higher energy.".format(_fmt_ratio(production_ratio)),
            _confidence_from_gap(production_ratio, 1.0),
        )
        return

    energy_rise = energy_ratio - 1.0
    production_rise = production_ratio - 1.0
    share = (production_rise / energy_rise) if energy_rise > 0 else 0.0

    evidence.append(
        "INFERRED production rise explains {0:.0%} of the energy rise".format(share)
    )

    if energy_rise > 0 and share >= PRODUCTION_EXPLANATION_SHARE:
        _set(
            hypothesis,
            STATUS_SUPPORTED,
            evidence,
            "Production rose {0:+.1%}, accounting for {1:.0%} of the energy "
            "rise; demand change plausibly explains this anomaly.".format(
                production_rise, share
            ),
            _confidence_from_share(share),
        )
    elif production_rise <= 0:
        _set(
            hypothesis,
            STATUS_REJECTED,
            evidence,
            "Production was flat to lower while energy rose; not a demand effect.",
            _confidence_from_gap(production_ratio, 1.0),
        )
    else:
        _set(
            hypothesis,
            STATUS_INCONCLUSIVE,
            evidence,
            "Production rose {0:+.1%} but accounts for only {1:.0%} of the "
            "energy rise, so it is a minor contributor rather than the "
            "explanation.".format(production_rise, share),
            _confidence_from_share(share),
        )


def _judge_runtime(hypothesis, runtime_ratio, intensity_ratio):
    """Runtime rose while the machine's power draw stayed flat."""
    if runtime_ratio is None or intensity_ratio is None:
        _set(
            hypothesis,
            STATUS_INCONCLUSIVE,
            [],
            "Runtime or intensity comparison unavailable.",
            0.0,
        )
        return

    evidence = [
        "OBSERVED runtime ratio {0}".format(_fmt_ratio(runtime_ratio)),
        "OBSERVED intensity ratio {0}".format(_fmt_ratio(intensity_ratio)),
    ]

    if runtime_ratio >= MATERIAL_RISE:
        if intensity_ratio < MATERIAL_RISE:
            _set(
                hypothesis,
                STATUS_SUPPORTED,
                evidence,
                "Runtime rose {0:+.1%} while energy per runtime hour stayed "
                "flat ({1:+.1%}), which indicates more operating time rather "
                "than a change in machine efficiency.".format(
                    runtime_ratio - 1.0, intensity_ratio - 1.0
                ),
                _confidence_from_gap(runtime_ratio, 1.0),
            )
        else:
            _set(
                hypothesis,
                STATUS_INCONCLUSIVE,
                evidence
                + [
                    "OBSERVED both runtime ({0:+.1%}) and intensity ({1:+.1%}) "
                    "rose".format(runtime_ratio - 1.0, intensity_ratio - 1.0)
                ],
                "Runtime and intensity both rose, so a scheduling change alone "
                "cannot account for the energy increase.",
                0.4,
            )
        return

    _set(
        hypothesis,
        STATUS_REJECTED,
        evidence,
        "Runtime was stable ({0:+.1%}); the machine did not run longer, so a "
        "scheduling increase is not the explanation.".format(runtime_ratio - 1.0),
        _confidence_from_gap(1.0, runtime_ratio),
    )


def _judge_equipment(hypothesis, intensity_ratio, efficiency_ratio, runtime_ratio):
    """Energy per runtime hour rose while runtime stayed flat."""
    if intensity_ratio is None:
        _set(
            hypothesis,
            STATUS_INCONCLUSIVE,
            [],
            "Intensity comparison unavailable.",
            0.0,
        )
        return

    evidence = [
        "OBSERVED intensity ratio {0}".format(_fmt_ratio(intensity_ratio)),
        "OBSERVED efficiency_factor ratio {0}".format(_fmt_ratio(efficiency_ratio)),
    ]
    if runtime_ratio is not None:
        evidence.append("OBSERVED runtime ratio {0}".format(_fmt_ratio(runtime_ratio)))

    if intensity_ratio < MATERIAL_RISE:
        _set(
            hypothesis,
            STATUS_REJECTED,
            evidence,
            "Energy per runtime hour was stable ({0:+.1%}); the machine was not "
            "working harder for the same time, so equipment degradation is not "
            "supported.".format(intensity_ratio - 1.0),
            _confidence_from_gap(1.0, intensity_ratio),
        )
        return

    supporting = "Energy per runtime hour rose {0:+.1%}".format(intensity_ratio - 1.0)
    if efficiency_ratio is not None and efficiency_ratio >= MATERIAL_RISE:
        evidence.append(
            "OBSERVED recorded efficiency_factor rose {0:+.1%}".format(
                efficiency_ratio - 1.0
            )
        )
        supporting += ", corroborated by the recorded efficiency_factor ({0:+.1%})".format(
            efficiency_ratio - 1.0
        )

    if runtime_ratio is not None and runtime_ratio < FLAT:
        supporting += (
            ", while runtime stayed flat ({0:+.1%})".format(runtime_ratio - 1.0)
        )

    _set(
        hypothesis,
        STATUS_SUPPORTED,
        evidence,
        supporting + ". More power was drawn for the same operating time, "
        "which points to an equipment or load-side condition.",
        _confidence_from_gap(intensity_ratio, 1.0),
    )


def _judge_maintenance(hypothesis, evidence):
    """Supported by an actual maintenance record overlapping the window."""
    records = evidence["maintenance_records"]
    if not records:
        _set(
            hypothesis,
            STATUS_INCONCLUSIVE,
            ["OBSERVED no maintenance records for this machine in the window"],
            "No maintenance activity overlaps the anomaly window, so there is "
            "no direct evidence either way.",
            0.0,
        )
        return

    items = [
        "OBSERVED {0} on {1} ({2}): {3}".format(
            record["maintenance_type"],
            record["date"],
            record["status"],
            record["notes"],
        )
        for record in records
    ]
    overdue = [r for r in records if r["status"] == "overdue"]

    if overdue:
        _set(
            hypothesis,
            STATUS_SUPPORTED,
            items,
            "{0} overdue maintenance record(s) overlap the anomaly window, "
            "which is consistent with an unresolved equipment condition.".format(
                len(overdue)
            ),
            0.7,
        )
    else:
        _set(
            hypothesis,
            STATUS_INCONCLUSIVE,
            items,
            "Maintenance activity overlaps the window but none of it is marked "
            "overdue, so it is context rather than a demonstrated cause.",
            0.4,
        )


def _judge_historical(hypothesis, evidence):
    """Supported by a prior verified incident for the same machine."""
    incidents = evidence["historical_incidents"]
    if not incidents:
        _set(
            hypothesis,
            STATUS_INCONCLUSIVE,
            ["OBSERVED no prior incidents recorded for this machine"],
            "This machine has no incident history, so there is no precedent to "
            "compare against.",
            0.0,
        )
        return

    items = [
        "OBSERVED {0} ({1}): {2} -> {3}".format(
            incident["incident_id"],
            incident["date"],
            incident["symptom"],
            incident["intervention"],
        )
        for incident in incidents
    ]
    verified = [incident for incident in incidents if incident["verified"]]

    if verified:
        _set(
            hypothesis,
            STATUS_SUPPORTED,
            items,
            "{0} verified prior incident(s) exist for this machine; the earlier "
            "intervention is usable as precedent for the next step.".format(
                len(verified)
            ),
            0.6,
        )
    else:
        _set(
            hypothesis,
            STATUS_INCONCLUSIVE,
            items,
            "Prior incidents exist but none are verified, so they are weak "
            "precedent.",
            0.3,
        )


def select_root_cause(hypotheses: list, evidence: dict) -> dict | None:
    """Pick the strongest supported hypothesis, or ``None`` if there is none.

    A hypothesis must be ``supported`` to be eligible. Ranking is by
    confidence first, because confidence already encodes how far the measured
    ratios sat from the flat band. ``_cause_rank`` only breaks ties, ordering
    measured-metric explanations ahead of contextual ones, which are backed by
    record presence rather than by a numeric comparison.
    """
    supported = [h for h in hypotheses if h["status"] == STATUS_SUPPORTED]
    if not supported:
        return None

    ranked = sorted(
        supported,
        key=lambda h: (-h["confidence"], _cause_rank(h["name"])),
    )
    winner = ranked[0]

    return {
        "category": winner["category"],
        "statement": _root_cause_statement(winner, evidence),
        "confidence": winner["confidence"],
        "supporting_evidence": list(winner["evidence"]),
        "recommended_next_action": next_action_for(winner["name"], evidence),
    }


def _cause_rank(name: str) -> int:
    """Lower rank wins; measured explanations beat contextual ones."""
    order = [
        HYPOTHESIS_EQUIPMENT,
        HYPOTHESIS_RUNTIME,
        HYPOTHESIS_PRODUCTION,
        HYPOTHESIS_MAINTENANCE,
        HYPOTHESIS_HISTORICAL,
    ]
    return order.index(name) if name in order else len(order)


def _root_cause_statement(hypothesis, evidence: dict) -> str:
    """A probable-cause sentence, deliberately hedged."""
    machine = evidence["machine"].get("machine_name", evidence["machine"]["machine_id"])
    window = evidence["window"]
    span = "{0} to {1}".format(window["start_date"], window["end_date"])

    if hypothesis["name"] == HYPOTHESIS_EQUIPMENT:
        return (
            "Probable equipment efficiency degradation on {0} between {1}: the "
            "machine drew more power for the same operating time.".format(
                machine, span
            )
        )
    if hypothesis["name"] == HYPOTHESIS_RUNTIME:
        return (
            "Probable operational/scheduling change on {0} between {1}: the "
            "machine ran substantially longer at its normal power draw.".format(
                machine, span
            )
        )
    if hypothesis["name"] == HYPOTHESIS_PRODUCTION:
        return (
            "Probable production-driven increase on {0} between {1}: output rose "
            "in step with energy use.".format(machine, span)
        )
    if hypothesis["name"] == HYPOTHESIS_MAINTENANCE:
        return (
            "Probable maintenance-related condition on {0} between {1}: "
            "maintenance records overlap the anomaly window.".format(machine, span)
        )
    return (
        "Probable recurrence of a previously seen condition on {0} between "
        "{1}.".format(machine, span)
    )


def next_action_for(hypothesis_name: str, evidence: dict) -> str:
    """The action this phase recommends; no intervention is performed."""
    machine = evidence["machine"].get("machine_name", evidence["machine"]["machine_id"])
    window = evidence["window"]
    span = "{0} to {1}".format(window["start_date"], window["end_date"])

    if hypothesis_name == HYPOTHESIS_EQUIPMENT:
        return (
            "Inspect {0} load-side equipment over {1} (filters, heating "
            "elements, seals and leaks) and confirm whether power draw per "
            "runtime hour returns to baseline after service.".format(machine, span)
        )
    if hypothesis_name == HYPOTHESIS_RUNTIME:
        return (
            "Compare the {0} operating schedule against production demand for "
            "{1} and confirm whether extended runtime was planned.".format(
                machine, span
            )
        )
    if hypothesis_name == HYPOTHESIS_PRODUCTION:
        return (
            "Reconcile the {0} output increase for {1} against order intake "
            "before treating the energy rise as avoidable.".format(machine, span)
        )
    if hypothesis_name == HYPOTHESIS_MAINTENANCE:
        overdue = [
            record
            for record in evidence["maintenance_records"]
            if record["status"] == "overdue"
        ]
        if overdue:
            return (
                "Complete the overdue {0} maintenance recorded on {1}, then "
                "re-measure energy per runtime hour.".format(
                    overdue[0]["maintenance_type"], overdue[0]["date"]
                )
            )
        return "Review maintenance history for {0} around {1}.".format(machine, span)

    verified = [i for i in evidence["historical_incidents"] if i["verified"]]
    if verified:
        return (
            "Apply the intervention from verified incident {0} ({1}) as a "
            "starting point, then verify the effect.".format(
                verified[0]["incident_id"], verified[0]["intervention"]
            )
        )
    return "Gather further evidence for {0} before acting.".format(machine)


def _next_action_when_inconclusive(evidence: dict) -> str:
    machine = evidence["machine"].get("machine_name", evidence["machine"]["machine_id"])
    return (
        "Evidence is insufficient to name a cause for {0}. Extend the "
        "observation window or collect additional readings before acting.".format(
            machine
        )
    )


def _set(hypothesis: dict, status: str, evidence: list, reason: str, confidence: float) -> None:
    hypothesis["status"] = status
    hypothesis["evidence"] = list(evidence)
    hypothesis["reason"] = reason
    hypothesis["confidence"] = round(min(max(confidence, 0.0), CONFIDENCE_CEILING), 3)


def _confidence_from_gap(observed_ratio, reference_ratio) -> float:
    """Confidence from how far a measured ratio sits from the flat band.

    At the flat band the verdict is weak; well outside it the verdict is strong.
    Uses only measured distance, so identical measurements always score
    identically regardless of which machine produced them.
    """
    if observed_ratio is None or reference_ratio is None:
        return 0.0
    distance = abs(observed_ratio - reference_ratio)
    if distance >= MATERIAL_RISE - 1.0:
        span = 0.25
    else:
        span = max(MATERIAL_RISE - 1.0, 1e-9)
    scaled = min(distance / span, 1.0)
    return CONFIDENCE_FLOOR + (CONFIDENCE_CEILING - CONFIDENCE_FLOOR) * scaled


def _confidence_from_share(share: float) -> float:
    """Confidence from how much of the energy rise production accounts for."""
    scaled = min(max(share / PRODUCTION_EXPLANATION_SHARE, 0.0), 1.0)
    return CONFIDENCE_FLOOR + (CONFIDENCE_CEILING - CONFIDENCE_FLOOR) * scaled


def _fmt_ratio(ratio) -> str:
    if ratio is None:
        return "n/a"
    return "{0:.3f}x".format(ratio)


def _ratio_text(comparison: dict) -> str:
    ratio = comparison.get("ratio")
    change = comparison.get("change_pct")
    if ratio is None or change is None:
        return "n/a"
    return "{0:+.1f}%".format(change)


def _verdict_summary(hypotheses: list) -> str:
    parts = [
        "{0}={1}".format(h["name"], h["status"]) for h in hypotheses
    ]
    return "; ".join(parts)