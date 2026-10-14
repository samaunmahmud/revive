import { useEffect, useState } from 'react'
import { EVENT_KINDS } from './api'

// How long each event stays "fresh" before the next one is shown. The agent
// really finishes a whole photo in well under a second; pacing the display lets
// a person follow the reasoning. The ms timings shown in each step are real.
const PACE = { queued: 0, start: 500, step: 650, human: 350, question: 650, notice: 300, done: 0, error: 0 }

/** Subscribe to a job's Server-Sent Events. Returns the events revealed so far. */
export function useJobStream(jobId) {
  // Events are stored with the job they belong to, so switching jobs shows an
  // empty list straight away without resetting state inside the effect.
  const [stream, setStream] = useState({ jobId: null, events: [] })

  useEffect(() => {
    if (!jobId) return
    const queue = []
    let timer = null

    const pump = () => {
      const next = queue.shift()
      if (!next) { timer = null; return }
      setStream((s) => (s.jobId === jobId ? { jobId, events: [...s.events, next] } : { jobId, events: [next] }))
      timer = setTimeout(pump, PACE[next.kind] ?? 300)
    }

    // EventSource reconnects by itself and sends Last-Event-ID, so the server
    // resumes where we left off instead of replaying everything.
    const es = new EventSource(`/api/jobs/${jobId}/events`)
    for (const kind of EVENT_KINDS) {
      es.addEventListener(kind, (msg) => {
        queue.push({ kind, data: JSON.parse(msg.data) })
        if (!timer) pump()
      })
    }
    es.addEventListener('end', () => es.close())

    return () => { es.close(); clearTimeout(timer) }
  }, [jobId])

  return { events: stream.jobId === jobId ? stream.events : [] }
}
