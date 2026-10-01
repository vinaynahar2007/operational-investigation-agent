import { useEffect, useState } from 'react'
import { fetchHealth, fetchMachines, runInvestigation } from './api.js'
import { AnomalyCard, EvidenceCard, ProgressPanel } from './components/panels.jsx'
import {
  HypothesesCard,
  InterventionCard,
  LlmCard,
  MemoryCard,
  RootCauseCard,
  VerificationCard,
} from './components/panels2.jsx'
import { Badge } from './components/ui.jsx'

export default function App() {
  const [machines, setMachines] = useState([])
  // M04 by default: it demonstrates the strongest complete investigation.
  const [machineId, setMachineId] = useState('M04')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [health, setHealth] = useState(null)

  useEffect(() => {
    fetchMachines()
      .then((data) => setMachines(data.machines || []))
      .catch(() => setMachines([]))
    fetchHealth()
      .then(setHealth)
      .catch(() => setHealth(null))
  }, [])

  async function onInvestigate() {
    setLoading(true)
    setError(null)
    try {
      const data = await runInvestigation(machineId)
      setResult(data)
    } catch (err) {
      setError(err.message || 'Investigation failed.')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Operational Investigation Agent</h1>
          <p className="tagline">AI-assisted operational root-cause investigation</p>
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

      <section className="controls">
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
        <button className="primary" onClick={onInvestigate} disabled={loading}>
          {loading ? 'Investigating…' : 'Investigate'}
        </button>
        {result ? (
          <button className="ghost" onClick={() => setResult(null)} disabled={loading}>
            Clear
          </button>
        ) : null}
      </section>

      {error ? (
        <div className="error-box">
          <strong>Investigation failed.</strong>
          <p>{error}</p>
        </div>
      ) : null}

      {!result && !error && !loading ? (
        <div className="empty-state">
          <p>
            Select a machine and run an investigation. All numbers, verdicts and
            conclusions on this page are computed by the Python backend; the UI
            only displays them.
          </p>
        </div>
      ) : null}

      {result ? (
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