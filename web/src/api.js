// Thin wrappers around the FastAPI endpoints (see server/app.py).

async function json(res) {
  const body = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`)
  return body
}

export const getConfig = () => fetch('/api/config').then(json)
export const getSamples = () => fetch('/api/samples').then(json)

export function startJob({ file, sample, planner }) {
  const form = new FormData()
  if (file) form.append('file', file)
  if (sample) form.append('sample', sample)
  form.append('planner', planner)
  return fetch('/api/jobs', { method: 'POST', body: form }).then(json)
}

export function answerJob(jobId, answers) {
  return fetch(`/api/jobs/${jobId}/answer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ answers }),
  }).then(json)
}

export const EVENT_KINDS = ['queued', 'start', 'step', 'human', 'question', 'notice', 'done', 'error']
