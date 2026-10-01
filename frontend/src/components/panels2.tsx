import type { InvestigationResult, MemoryMatch } from '../types'
import { Badge, Card, Num, Pct, Row, StatusBadge } from './ui'

export function HypothesesCard({ result }: { result: InvestigationResult }) {
  const hypotheses = result.hypotheses || []
  return (
    <Card title="Hypotheses" subtitle="Each candidate and the verdict the evaluator reached." tag={{ text: 'inferred', tone: 'info' }}>
      {hypotheses.length === 0 ? (
        <p className="muted">No hypotheses were evaluated.</p>
      ) : (
        <ul className="hyp-list">
          {hypotheses.map((h) => (
            <li key={h.name} className="hyp">
              <div className="hyp-head">
                <span className="hyp-name">{humanise(h.name)}</span>
                <StatusBadge status={h.status} />
                <span className="hyp-conf">confidence <Num value={h.confidence} digits={2} /></span>
              </div>
              <p className="hyp-reason">{h.reason}</p>
              {h.evidence && h.evidence.length > 0 ? (
                <details className="hyp-evidence">
                  <summary>Evidence ({h.evidence.length})</summary>
                  <ul className="plain-list">
                    {h.evidence.map((item, i) => <li key={i}>{item}</li>)}
                  </ul>
                </details>
              ) : null}
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}

export function RootCauseCard({ result }: { result: InvestigationResult }) {
  const rc = result.root_cause
  if (!rc) {
    return (
      <Card title="Probable Root Cause" tag={{ text: 'inconclusive', tone: 'warn' }}>
        <p className="muted">
          No hypothesis was supported, so no root cause is asserted. This is a deliberate
          refusal rather than a missing value.
        </p>
      </Card>
    )
  }
  return (
    <Card title="Probable Root Cause" tag={{ text: rc.category, tone: 'good' }}>
      <p className="statement">{rc.statement}</p>
      <Row label="Category" value={rc.category} />
      <Row label="Confidence" value={<Num value={rc.confidence} digits={2} />} />
      <h3 className="sub-head">Supporting evidence</h3>
      <ul className="plain-list">
        {(rc.supporting_evidence || []).map((item, i) => <li key={i}>{item}</li>)}
      </ul>
      <h3 className="sub-head">Recommended next action</h3>
      <p className="action">{rc.recommended_next_action || result.recommended_next_action}</p>
    </Card>
  )
}

export function MemoryCard({ result }: { result: InvestigationResult }) {
  const matches: MemoryMatch[] = result.memory_matches || []
  return (
    <Card
      title="Operational Memory"
      subtitle="Prior investigations retrieved by deterministic field matching."
      tag={{ text: `${matches.length} match${matches.length === 1 ? '' : 'es'}`, tone: matches.length ? 'info' : 'muted' }}
    >
      <p className="authority">Historical precedent only. Current evidence remains authoritative.</p>
      {matches.length === 0 ? (
        <p className="muted">No similar historical investigation found.</p>
      ) : (
        matches.map((match) => (
          <div key={match.id} className="mem-case">
            <div className="mem-head">
              <span className="strong">Case #{match.id}</span>
              <Badge tone="info">similarity {match.score}</Badge>
              {match.case?.machine_id ? <span>{match.case.machine_id}</span> : null}
            </div>
            <Row label="Root cause" value={match.case?.root_cause_category} />
            <Row label="Intervention" value={match.case?.intervention_description} />
            <Row
              label="Verification"
              value={
                <span>
                  <StatusBadge status={match.case?.verification_status} />
                  {match.case?.simulation ? <Badge tone="warn"> simulated</Badge> : null}
                </span>
              }
            />
            <Row label="Why considered similar" value={(match.reason || []).join('; ')} />
          </div>
        ))
      )}
      <div className="mem-store">
        <h3 className="sub-head">Stored for future investigations</h3>
        {result.memory_record_id ? (
          <p>
            Stored as investigation <span className="strong">#{result.memory_record_id}</span>{' '}
            <Badge tone="warn">simulation: yes</Badge>{' '}
            <span className="muted">a simulated outcome, not a measured result.</span>
          </p>
        ) : (
          <p className="muted">This investigation was not stored in operational memory.</p>
        )}
      </div>
    </Card>
  )
}

export function InterventionCard({ result }: { result: InvestigationResult }) {
  const iv = result.intervention
  if (!iv) {
    return <Card title="Intervention"><p className="muted">No intervention was simulated.</p></Card>
  }
  return (
    <Card
      title="Intervention"
      subtitle="Proposed action for the probable root cause."
      tag={{ text: 'simulated', tone: 'warn' }}
    >
      <Row label="Type" value={iv.type} />
      <p className="statement">{iv.description}</p>
      <Row label="Baseline energy" value={<><Num value={iv.baseline_energy} /> kWh/day</>} />
      <Row label="Projected energy" value={<><Num value={iv.projected_energy} /> kWh/day</>} />
      <Row label="Projected reduction" value={<Pct value={iv.projected_reduction_percent} digits={2} />} />
      <p className="disclaimer">
        SIMULATED: this is an arithmetic projection. No machine was physically changed.
      </p>
    </Card>
  )
}

export function VerificationCard({ result }: { result: InvestigationResult }) {
  const v = result.verification
  if (!v) {
    return <Card title="Verification"><p className="muted">Nothing was verified.</p></Card>
  }
  return (
    <Card
      title="Verification"
      subtitle="Arithmetic check of the projection against a threshold."
      tag={{ text: v.simulation ? 'simulated' : 'measured', tone: 'warn' }}
    >
      <div className="verify-headline">
        <StatusBadge status={v.status} />
        <span className="verify-label">
          {v.status === 'PASS' ? 'Projection meets threshold' : 'Projection below threshold'}
        </span>
      </div>
      <Row label="Observed reduction" value={<Pct value={v.observed_reduction_percent} digits={2} />} />
      <Row label="Assumed reduction" value={<Pct value={v.expected_reduction_percent} digits={2} />} />
      <Row label="Threshold" value={<Pct value={v.threshold_percent} digits={2} />} />
      <Row label="Reason" value={v.reason} />
      <p className="disclaimer">
        SIMULATED: this verifies the arithmetic only. No machine was measured.
      </p>
    </Card>
  )
}

export function LlmCard({ result }: { result: InvestigationResult }) {
  const llm = result.llm_reasoning
  if (!llm) return null
  const traceEntry = result.trace.find((t) => t.step === 'LLM_REASONING')
  const status = traceEntry ? traceEntry.status : 'skipped'
  return (
    <Card
      title="LLM Reasoning"
      subtitle="Optional reasoning layer. Deterministic results stand without it."
      tag={{ text: status, tone: status === 'complete' ? 'good' : 'muted' }}
    >
      <Row label="Provider" value={traceEntry?.provider} />
      <Row label="Model" value={traceEntry?.model} />
      <Row label="Status" value={<StatusBadge status={status} />} />
      {llm.available ? (
        <>
          <h3 className="sub-head">Model reasoning</h3>
          <p className="statement">{llm.reasoning_summary}</p>
          {llm.root_cause_explanation ? (
            <p className="reason-line">{llm.root_cause_explanation}</p>
          ) : null}
        </>
      ) : (
        <p className="muted">
          {llm.reason || 'LLM unavailable.'} Falling back to the deterministic investigator.
        </p>
      )}
      {(llm.conflicts || []).length > 0 ? (
        <>
          <h3 className="sub-head">Conflicts (deterministic verdict kept)</h3>
          <ul className="plain-list">
            {llm.conflicts.map((c, i) => (
              <li key={i}>
                {c.hypothesis}: model said <Badge tone="bad">{c.model_status}</Badge>, evidence says{' '}
                <Badge tone="good">{c.deterministic_status}</Badge>
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </Card>
  )
}

function humanise(name: string): string {
  return String(name || '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
}
