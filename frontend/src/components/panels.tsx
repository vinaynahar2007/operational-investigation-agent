import type { InvestigationResult, TraceEntry } from '../types'
import { Badge, Card, Num, Pct, Row, StatusBadge, type Tone } from './ui'

/** Investigation stages exactly as the backend reports them, in order. */
export function ProgressPanel({ trace }: { trace: TraceEntry[] }) {
  if (!trace || trace.length === 0) {
    return <Card title="Investigation Progress"><p className="muted">No trace returned.</p></Card>
  }
  return (
    <Card title="Investigation Progress" subtitle="Stage order and status as returned by the investigator.">
      <ol className="stages">
        {trace.map((entry, index) => (
          <li key={entry.step} className={`stage stage-${entry.status}`}>
            <span className="stage-index">{String(index + 1).padStart(2, '0')}</span>
            <span className="stage-name">{entry.step}</span>
            <StatusBadge status={entry.status} />
            {entry.provider ? <span className="stage-meta">{entry.provider}</span> : null}
            <p className="stage-summary">{entry.summary}</p>
          </li>
        ))}
      </ol>
    </Card>
  )
}

export function AnomalyCard({ result }: { result: InvestigationResult }) {
  const a = result.anomaly
  if (!a) {
    return (
      <Card title="Anomaly">
        <p className="muted">No anomaly was detected for this machine.</p>
      </Card>
    )
  }
  return (
    <Card
      title="Anomaly"
      subtitle={`${a.start_date} → ${a.end_date}`}
      tag={{ text: a.severity, tone: severityTone(a.severity) }}
    >
      <Row label="Machine" value={a.machine_id} />
      <Row label="Metric" value={a.metric} />
      <Row label="Anomaly type" value={a.anomaly_type} />
      <Row label="Magnitude" value={<Pct value={a.magnitude_pct} />} />
      <Row label="Peak magnitude" value={<Pct value={a.peak_magnitude_pct} />} />
      <Row label="Severity" value={<StatusBadge status={a.severity} />} />
      <Row label="Confidence" value={<Num value={a.confidence} digits={2} />} />
      <Row label="Days" value={a.days} />
    </Card>
  )
}

function severityTone(severity: string): Tone {
  if (severity === 'high') return 'bad'
  if (severity === 'medium') return 'warn'
  return 'muted'
}

export function EvidenceCard({ result }: { result: InvestigationResult }) {
  const evidence = result.evidence
  if (!evidence) return <Card title="Evidence"><p className="muted">No evidence collected.</p></Card>

  const m = evidence.metrics
  const production = evidence.production
  const machine = evidence.machine
  const maintenance = evidence.maintenance_records
  const notes = evidence.technician_notes
  const incidents = evidence.historical_incidents

  return (
    <Card title="Evidence" subtitle="Deterministic measurements from the dataset." tag={{ text: 'observed', tone: 'info' }}>
      <div className="metric-grid">
        <Metric label="Energy" change={m.energy_kwh.change_pct} note={`${fmt(m.energy_kwh.window_value)} kWh/day`} />
        <Metric label="Intensity" change={m.intensity.change_pct} note="kWh per runtime hour" />
        <Metric label="Runtime" change={m.runtime_h.change_pct} note={`${fmt(m.runtime_h.window_value)} h/day`} />
        <Metric label="Efficiency factor" change={m.efficiency_factor.change_pct} note="recorded factor" />
        <Metric label="Production" change={production.change_pct} note={`${fmt(production.window_mean_units)} units/day`} />
      </div>

      <div className="baseline-note">
        Compared against the preceding {evidence.baseline_window.days}-day baseline (
        {evidence.baseline_window.start_date} → {evidence.baseline_window.end_date}).
      </div>

      <h3 className="sub-head">Machine</h3>
      {machine.found ? (
        <>
          <Row label="Name" value={machine.machine_name} />
          <Row label="Category" value={machine.category} />
          <Row label="Rated power" value={`${fmt(machine.rated_kw)} kW`} />
        </>
      ) : (
        <p className="muted">This machine is not present in the dataset.</p>
      )}

      <h3 className="sub-head">Maintenance in window ({maintenance.length})</h3>
      {maintenance.length === 0 ? (
        <p className="muted">No maintenance records overlap this window.</p>
      ) : (
        <ul className="plain-list">
          {maintenance.map((r, i) => (
            <li key={i}>
              <StatusBadge status={r.status === 'overdue' ? 'failed' : 'skipped'} />
              <span className="strong"> {r.maintenance_type}</span> — {r.date} — {r.notes}
            </li>
          ))}
        </ul>
      )}

      <h3 className="sub-head">Technician notes in window ({notes.length})</h3>
      {notes.length === 0 ? (
        <p className="muted">No technician notes in this window.</p>
      ) : (
        <ul className="plain-list">
          {notes.map((n, i) => <li key={i}>{n.date} — {n.note}</li>)}
        </ul>
      )}

      <h3 className="sub-head">Prior incidents ({incidents.length})</h3>
      {incidents.length === 0 ? (
        <p className="muted">No prior incidents for this machine.</p>
      ) : (
        <ul className="plain-list">
          {incidents.map((inc, i) => (
            <li key={i}>
              {inc.incident_id} — {inc.date} — {inc.symptom} → {inc.intervention}
              {inc.within_window ? <Badge tone="warn"> in window</Badge> : null}
            </li>
          ))}
        </ul>
      )}
    </Card>
  )
}

interface MetricProps {
  label: string
  change: number | null
  note: string
}

function Metric({ label, change, note }: MetricProps) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${toneForChange(change)}`}><Pct value={change} /></div>
      <div className="metric-note">{note}</div>
    </div>
  )
}

function toneForChange(change: number | null): Tone {
  if (change === null) return 'muted'
  if (change > 1) return 'bad'
  if (change > 0) return 'warn'
  return 'good'
}

function fmt(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—'
  return Number(value).toFixed(2)
}
