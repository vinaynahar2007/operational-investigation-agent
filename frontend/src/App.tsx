import { useEffect, useState } from 'react'
import { fetchHealth, fetchMachines, runInvestigation } from './api'
import { Overview } from './components/overview'
import { AnomalyCard, EvidenceCard, ProgressPanel } from './components/panels'
import {
  HypothesesCard,
  InterventionCard,
  LlmCard,
  MemoryCard,
  RootCauseCard,
  VerificationCard,
} from './components/panels2'
import { Badge } from './components/ui'
import type { HealthResponse, InvestigationResult, Machine } from './types'

type View = 'overview' | 'investigation'

export default function App() {
  const [machines, setMachines] = useState<Machine[]>([])
  const [machineId, setMachineId] = useState('M04')
  const [result, setResult] = useState<InvestigationResult | null>(null)
  const [results, setResults] = useState<Record<string, InvestigationResult>>({})
  const [view, setView] = useState<View>('overview')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)

  useEffect(() => {
    fetchMachines()
      .then((data) => setMachines(data.machines || []))
      .catch(() => setMachines([]))
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  async function onInvestigate(targetMachineId: string = machineId): Promise<void> {
    setLoading(true)
    setError(null)
    setMachineId(targetMachineId)
    try {
      const data = await runInvestigation(targetMachineId)
      setResult(data)
      setResults((current) => ({ ...current, [data.machine_id]: data }))
      setView('investigation')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Investigation failed.')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand-lockup">
          <span className="brand-mark">OI</span>
          <div>
            <h1>Operational Intelligence</h1>
            <p className="tagline">Evidence-led energy operations</p>
          </div>
        </div>
        <div className="health">
          {health ? (
            <>
              <Badge tone={health.llm_configured ? 'good' : 'muted'}>
                backend ok
              </Badge>
              <Badge tone="muted">
                llm {health.llm_configured ? 'configured' : 'not configured'}
              </Badge>
            </>
          ) : (
            <Badge tone="bad">backend unreachable</Badge>
          )}
        </div>
      </header>

      <nav className="app-nav" aria-label="Primary navigation">
        <button className={view === 'overview' ? 'nav-active' : ''} onClick={() => setView('overview')}>
          Overview
        </button>
        <button className={view === 'investigation' ? 'nav-active' : ''} onClick={() => setView('investigation')}>
          Investigation workspace
        </button>
      </nav>

      {view === 'investigation' ? <section className="controls">
        <label htmlFor="machine">Machine</label>
        <select
          id="machine"
          value={machineId}
          onChange={(e) => setMachineId(e.target.value)}
        >
          {machines.length === 0 ? (
            <option value={machineId}>{machineId}</option>
          ) : (
            machines
              .filter((m) => ['M03', 'M04', 'M05'].includes(m.machine_id))
              .map((m) => (
                <option key={m.machine_id} value={m.machine_id}>
                  {m.machine_id} — {m.machine_name}
                </option>
              ))
          )}
        </select>
        <button className="primary" onClick={() => onInvestigate()} disabled={loading}>
          {loading ? 'Investigating…' : 'Investigate'}
        </button>
        {result ? (
          <button className="ghost" onClick={() => setResult(null)} disabled={loading}>
            Clear
          </button>
        ) : null}
      </section> : null}

      {error ? (
        <div className="error-box">
          <strong>Investigation failed.</strong>
          <p>{error}</p>
        </div>
      ) : null}

      {view === 'overview' && !error ? (
        <Overview
          machines={machines}
          results={results}
          onSelectMachine={onInvestigate}
          onOpenInvestigation={(id) => {
            setMachineId(id)
            setResult(results[id])
            setView('investigation')
          }}
        />
      ) : null}

      {view === 'investigation' && !result && !error && !loading ? (
        <div className="empty-state">
          <p>
            Select a machine and run an investigation. All numbers, verdicts and
            conclusions on this page are computed by the Python backend; the UI
            only displays them.
          </p>
        </div>
      ) : null}

      {view === 'investigation' && result ? (
        <main className="grid">
          <div className="col col-wide">
            <ProgressPanel trace={result.trace} />
            <EvidenceCard result={result} />
            <HypothesesCard result={result} />
            <LlmCard result={result} />
          </div>
          <div className="col col-narrow">
            <AnomalyCard result={result} />
            <RootCauseCard result={result} />
            <InterventionCard result={result} />
            <VerificationCard result={result} />
            <MemoryCard result={result} />
          </div>
        </main>
      ) : null}

      <footer className="app-footer">
        Investigation logic runs entirely in Python. Intervention and verification
        figures are SIMULATED projections, not measured real-world results.
      </footer>
    </div>
  )
}
