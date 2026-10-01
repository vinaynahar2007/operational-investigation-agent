/**
 * Type definitions for the Python backend's JSON contract.
 *
 * These types describe the shapes returned by `app/api.py` exactly as they are
 * produced by the investigator. The frontend renders these fields verbatim;
 * nothing here computes or transforms a value, and no number on screen
 * originates in TypeScript.
 */

// --- shared unions --------------------------------------------------------

/** Investigator status values (see app/agents/investigator.py). */
export type InvestigationStatus =
  | 'complete'
  | 'no_anomaly'
  | 'inconclusive'
  | 'failed'
  | 'verification_failed'
  | 'pending'

/** Anomaly severity bands assigned by the deterministic detector. */
export type Severity = 'high' | 'medium' | 'low'

/** Hypothesis verdicts from the deterministic evaluator. */
export type VerdictStatus = 'supported' | 'rejected' | 'inconclusive'

/** Trace entry statuses used by the investigator. */
export type TraceStatus = 'complete' | 'failed' | 'skipped'

/** Arithmetic verification outcomes (app/tools/verification.py). */
export type VerificationStatus = 'PASS' | 'FAIL'

// --- machines and health --------------------------------------------------

export interface Machine {
  machine_id: string
  machine_name: string
  category: string
}

export interface MachinesResponse {
  machines: Machine[]
}

export interface HealthResponse {
  status: string
  llm_configured: boolean
  llm_provider: string
}

// --- anomaly --------------------------------------------------------------

/** One finding from the deterministic detector. */
export interface AnomalyFinding {
  machine_id: string
  start_date: string
  end_date: string
  metric: string
  anomaly_type: string
  ratio: number
  magnitude_pct: number
  peak_magnitude_pct: number
  severity: Severity
  confidence: number
  days: number
  supporting_metrics: string[]
  baseline_value: number
  mean_intensity_ratio: number | null
  mean_runtime_ratio: number | null
  average_intensity: number | null
  average_runtime_h: number | null
  average_energy_kwh: number | null
  average_efficiency_factor: number | null
  production_change_pct: number | null
}

/** The investigated anomaly window, plus any further windows found. */
export interface Anomaly extends AnomalyFinding {
  additional_windows?: AnomalyFinding[]
}

// --- evidence -------------------------------------------------------------

/**
 * Window vs baseline comparison for a single metric. A null means the dataset
 * did not provide a comparable value; the UI shows those as a dash.
 */
export interface MetricComparison {
  window_value: number | null
  baseline_value: number | null
  ratio: number | null
  change_pct: number | null
}

export interface MetricComparisons {
  energy_kwh: MetricComparison
  runtime_h: MetricComparison
  intensity: MetricComparison
  efficiency_factor: MetricComparison
}

export interface ProductionComparison {
  window_mean_units: number | null
  baseline_mean_units: number | null
  ratio: number | null
  change_pct: number | null
}

/** Machine metadata as a discriminated union around the `found` flag. */
export type MachineEvidence =
  | { machine_id: string; found: false }
  | {
      machine_id: string
      found: true
      machine_name: string
      category: string
      rated_kw: number
      baseline_runtime_h: number
    }

export interface WindowSpan {
  start_date: string
  end_date: string
  days: number
}

export interface MaintenanceRecord {
  date: string
  maintenance_type: string
  status: string
  notes: string
}

export interface TechnicianNote {
  date: string
  note: string
  recommended_followup: string
}

export interface HistoricalIncident {
  incident_id: string
  date: string
  symptom: string
  description: string
  intervention: string
  verified_energy_reduction: number | null
  verified: boolean
  within_window: boolean
}

export interface DataQuality {
  metrics_available: number
  metrics_expected: number
  sufficient: boolean
  machine_known: boolean
}

export interface Evidence {
  machine: MachineEvidence
  anomaly: Anomaly
  window: WindowSpan
  baseline_window: WindowSpan
  metrics: MetricComparisons
  production: ProductionComparison
  maintenance_records: MaintenanceRecord[]
  technician_notes: TechnicianNote[]
  historical_incidents: HistoricalIncident[]
  data_quality: DataQuality
}

// --- hypotheses, root cause, intervention, verification -------------------

export interface Hypothesis {
  name: string
  category: string
  status: VerdictStatus
  evidence: string[]
  reason: string
  confidence: number
}

export interface RootCause {
  category: string
  statement: string
  confidence: number
  supporting_evidence: string[]
  recommended_next_action: string
}

export interface Intervention {
  type: string
  description: string
  baseline_energy: number
  projected_energy: number
  projected_reduction_percent: number
  simulation: boolean
}

export interface Verification {
  status: VerificationStatus
  baseline_energy: number
  projected_energy: number
  observed_reduction_percent: number
  expected_reduction_percent: number
  threshold_percent: number
  simulation: boolean
  reason: string
}

// --- optional reasoning layer ---------------------------------------------

/** One model/judgment assessment of a hypothesis (advisory only). */
export interface HypothesisAssessment {
  name: string
  status: VerdictStatus
  reason: string
  confidence: number
}

/** A recorded disagreement between the model and the measured evidence. */
export interface ReasoningConflict {
  hypothesis: string
  deterministic_status: VerdictStatus | null
  model_status: VerdictStatus | string
  resolution: string
  note: string
}

export interface LlmReasoning {
  available: boolean
  hypothesis_assessment: HypothesisAssessment[]
  root_cause_explanation: string
  recommended_intervention_explanation: string
  reasoning_summary: string
  conflicts: ReasoningConflict[]
  authoritative_source: string
  reason?: string
}

// --- operational memory ---------------------------------------------------

export interface MemoryCase {
  id: number
  created_at: string
  machine_id: string
  symptom: string
  anomaly_metric: string
  anomaly_magnitude: number
  root_cause_category: string
  root_cause_statement: string
  intervention_type: string
  intervention_description: string
  projected_reduction_percent: number
  verification_status: VerificationStatus
  confidence: number
  evidence_summary: string
  simulation: boolean
  source: string
}

export interface MemoryMatch {
  id: number
  score: number
  case: MemoryCase
  reason: string[]
}

// --- trace and the full result --------------------------------------------

export interface TraceEntry {
  step: string
  status: TraceStatus
  summary: string
  provider?: string
  model?: string
}

/** The complete investigation returned by POST /api/investigate. */
export interface InvestigationResult {
  machine_id: string
  question: string
  status: InvestigationStatus
  anomaly: Anomaly | null
  evidence: Evidence | null
  hypotheses: Hypothesis[]
  root_cause: RootCause | null
  intervention: Intervention | null
  verification: Verification | null
  llm_reasoning: LlmReasoning | null
  memory_matches: MemoryMatch[]
  memory_record_id: number | null
  recommended_next_action: string | null
  trace: TraceEntry[]
}

