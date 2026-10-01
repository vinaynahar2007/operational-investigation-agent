/**
 * Thin fetch wrapper around the Python backend.
 *
 * No investigation logic lives here. The frontend only requests a result and
 * renders the fields the backend returns; every number shown on screen comes
 * from the investigator, never from JavaScript.
 */

async function request(path, options) {
  const response = await fetch(path, options)
  let payload = null
  try {
    payload = await response.json()
  } catch {
    payload = null
  }
  if (!response.ok) {
    const detail =
      (payload && (payload.detail || payload.message)) || `HTTP ${response.status}`
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return payload
}

export function fetchHealth() {
  return request('/api/health')
}

export function fetchMachines() {
  return request('/api/machines')
}

export function runInvestigation(machineId) {
  return request('/api/investigate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ machine_id: machineId }),
  })
}