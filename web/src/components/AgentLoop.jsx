import { toolLabel } from '../labels'

const PHASES = [
  { id: 'perceive', label: 'Perceive' },
  { id: 'plan', label: 'Plan' },
  { id: 'act', label: 'Act' },
  { id: 'verify', label: 'Verify' },
  { id: 'ask', label: 'Ask' },
]

/** Which phases of the loop the latest event lit up, plus a one-line caption. */
function describe(event, done) {
  if (done) return { lit: [], caption: 'Loop finished', finished: true }
  if (!event) return { lit: ['perceive'], caption: 'Measuring the photo with OpenCV 5…' }
  const d = event.data
  switch (event.kind) {
    case 'start':
      return { lit: ['perceive'], caption: 'Measured six kinds of damage' }
    case 'step':
      return {
        lit: ['plan', 'act', 'verify'],
        caption: `${toolLabel(d.tool)} · ${d.kept ? 'kept: the re-measurement shows it helped' : 'rejected by the verifier'}`,
        sequence: true,
      }
    case 'question':
      return { lit: ['ask'], caption: `Not sure about ${toolLabel(d.about).toLowerCase()}, so it asks you` }
    case 'human':
      return { lit: ['plan'], caption: `Using your answer: ${d.approved ? 'go ahead' : 'leave it'}` }
    default:
      return { lit: ['plan'], caption: 'Planning the next step…' }
  }
}

/** Live strip showing where the agent is in its perceive → plan → act → verify → ask loop. */
export default function AgentLoop({ event, done, eventKey }) {
  const { lit, caption, sequence, finished } = describe(event, done)
  return (
    <div className={`agent-loop ${finished ? 'finished' : ''}`} aria-live="polite">
      <ol>
        {PHASES.map((p) => {
          const i = lit.indexOf(p.id)
          return (
            <li
              key={`${p.id}-${eventKey}`}
              className={i >= 0 ? 'lit' : ''}
              style={sequence && i >= 0 ? { animationDelay: `${i * 160}ms` } : undefined}
            >
              {p.label}
            </li>
          )
        })}
      </ol>
      <span className="agent-loop-caption" key={eventKey}>{caption}</span>
    </div>
  )
}
