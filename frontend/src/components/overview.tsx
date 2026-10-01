import type { InvestigationResult, Machine, Severity } from '../types'
import { Card, Pct, StatusBadge, type Tone } from './ui'

interface OverviewProps {
  machines: Machine[]
  results: Record<string, InvestigationResult>
  onSelectMachine: (machineId: string) => void
  onOpenInvestigation: (machineId: string) => void
}

export function Overview({
  machines,
  results,
  onSelectMachine,
  onOpenInvestigation,
}: OverviewProps) {
  const investigated = Object.values(results)
  const anomalies = investigated.flatMap((result) =>
    result.anomaly ? [{ result, anomaly: result.anomaly }] : [],
  )
  const critical = anomalies.filter((entry) => entry.anomaly.severity === 'high')
  const completed = investigated.filter((result) => result.status === 'complete')
  const assessedCount = investigated.length

  return (
    <main className="overview">
      <section className="hero-panel">
        <div>
          <p className="eyebrow">Plant operations / live workspace</p>
          <h2>See what needs attention before it becomes downtime.</h2>
          <p className="hero-copy">
            Investigate machines from one place, keep measured evidence visible, and move from signal to a defensible next action.
          </p>
        </div>
        <div className="hero-status">
          <span className="pulse-dot" />
          <span>Evidence engine ready</span>
        </div>
      </section>

      <section className="kpi-grid" aria-label="Operations summary">
        <Kpi label="Fleet" value={machines.length} detail={`${assessedCount} assessed this session`} tone="info" />
        <Kpi label="Active anomalies" value={anomalies.length} detail="From completed investigations" tone={anomalies.length ? 'warn' : 'good'} />
        <Kpi label="High severity" value={critical.length} detail="Requires priority review" tone={critical.length ? 'bad' : 'good'} />
        <Kpi label="Completed" value={completed.length} detail="Verified investigation flows" tone="good" />
      </section>

      <section className="overview-grid">
        <Card title="Machine health" subtitle="Select a machine to investigate or revisit its latest result." tag={{ text: `${machines.length} machines`, tone: 'info' }}>
          <div className="machine-grid">
            {machines.map((machine) => (
              <MachineCard
                key={machine.machine_id}
                machine={machine}
                result={results[machine.machine_id]}
                onClick={() => onSelectMachine(machine.machine_id)}
              />
            ))}
          </div>
        </Card>

        <Card title="Priority queue" subtitle="Latest assessed signals from this session.">
          {anomalies.length === 0 ? (
            <div className="queue-empty">
              <span className="empty-mark">--</span>
              <p>No assessed anomalies yet.</p>
              <span className="muted">Choose a machine above to start an evidence-backed investigation.</span>
            </div>
          ) : (
            <div className="queue-list">
              {anomalies
                .sort((a, b) => b.anomaly.magnitude_pct - a.anomaly.magnitude_pct)
                .map((entry) => (
                  <button className="queue-item" key={entry.result.machine_id} onClick={() => onOpenInvestigation(entry.result.machine_id)}>
                    <span className={`severity-bar severity-${entry.anomaly.severity}`} />
                    <span className="queue-main">
                      <strong>{entry.result.machine_id}</strong>
                      <span>{entry.anomaly.metric} anomaly</span>
                    </span>
                    <span className="queue-value"><Pct value={entry.anomaly.magnitude_pct} /></span>
                  </button>
                ))}
            </div>
          )}
        </Card>
      </section>

      <section className="overview-note">
        <span className="note-icon">i</span>
        <p>All values shown here come from the Python investigator. Unassessed machines are intentionally not scored.</p>
      </section>
    </main>
  )
}

interface KpiProps {
  label: string
  value: number
  detail: string
  tone: Tone
}

function Kpi({ label, value, detail, tone }: KpiProps) {
  return (
    <div className={`kpi kpi-${tone}`}>
      <span className="kpi-label">{label}</span>
      <strong className="kpi-value">{value}</strong>
      <span className="kpi-detail">{detail}</span>
    </div>
  )
}

/** Card tones: an anomaly severity band, or a health-based fallback. */
type MachineTone = Severity | 'good' | 'warn' | 'muted'

interface MachineCardProps {
  machine: Machine
  result: InvestigationResult | undefined
  onClick: () => void
}

function MachineCard({ machine, result, onClick }: MachineCardProps) {
  const anomaly = result?.anomaly
  const status = anomaly ? anomaly.severity : result ? result.status : 'unassessed'
  const healthy =
    result !== undefined &&
    (result.status === 'complete' || result.status === 'no_anomaly')
  const tone: MachineTone = anomaly
    ? anomaly.severity
    : result
      ? (healthy ? 'good' : 'warn')
      : 'muted'

  return (
    <button className={`machine-card machine-${tone}`} onClick={onClick}>
      <span className="machine-card-top">
        <span className="machine-id">{machine.machine_id}</span>
        <StatusBadge status={status} />
      </span>
      <strong className="machine-name">{machine.machine_name}</strong>
      <span className="machine-category">{machine.category.replace(/_/g, ' ')}</span>
      <span className="machine-card-bottom">
        {anomaly ? (
          <><span>{anomaly.metric} elevated</span><Pct value={anomaly.magnitude_pct} /></>
        ) : result ? (
          <><span>No anomaly detected</span><span>stable</span></>
        ) : (
          <><span>Awaiting review</span><span>open</span></>
        )}
      </span>
    </button>
  )
}
