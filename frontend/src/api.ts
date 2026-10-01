/**
 * Thin fetch wrapper around the Python backend.
 *
 * No investigation logic lives here. The frontend only requests a result and
 * renders the fields the backend returns; every number shown on screen comes
 * from the investigator, never from JavaScript.
 */
import type {
  HealthResponse,
  InvestigationResult,
  MachinesResponse,
} from './types'

/** Error bodies carry either a message string or a structured detail object. */
function detailFrom(payload: unknown, status: number): string {
  if (payload && typeof payload === 'object') {
    const record = payload as Record<string, unknown>
    const detail = record.detail ?? record.message
    if (typeof detail === 'string') return detail
    if (detail !== undefined) return JSON.stringify(detail)
  }
  return `HTTP ${status}`
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options)
  let payload: unknown = null
  try {
    payload = await response.json()
  } catch {
    payload = null
  }
  if (!response.ok) {
    throw new Error(detailFrom(payload, response.status))
  }
  // The backend is the source of truth for this shape; the cast happens once,
  // at the network boundary, after an OK status.
  return payload as T
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/api/health')
}

export function fetchMachines(): Promise<MachinesResponse> {
  return request<MachinesResponse>('/api/machines')
}

export function runInvestigation(machineId: string): Promise<InvestigationResult> {
  return request<InvestigationResult>('/api/investigate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ machine_id: machineId }),
  })
}
