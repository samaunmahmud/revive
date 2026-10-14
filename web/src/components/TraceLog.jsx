import { fmtMetric, metricLabel, toolLabel } from '../labels'

const REVEALED_NAMES = { noise_sigma: 'grain', scratch_coverage: 'scratches' }

function StepEntry({ s }) {
  const fallback = s.reason?.startsWith('[fallback')
  const before = s.measured_before?.[s.target]
  const after = s.measured_after?.[s.target]
  return (
    <>
      <div className="entry-head">
        <span className="entry-num">{s.step}</span>
        <span className="entry-tool">{toolLabel(s.tool)}</span>
        {s.image && <img className="entry-thumb" src={s.image} alt="" />}
        <span className={`verdict ${s.kept ? 'kept' : 'rejected'}`}>{s.kept ? 'kept' : 'rejected'}</span>
      </div>
      <p className="entry-reason">
        {fallback && <span className="badge fallback">Bedrock fallback</span>}
        {s.reason}
      </p>
      <div className="entry-facts">
        {Object.keys(s.params || {}).length > 0 && (
          <span className="fact">{Object.entries(s.params).map(([k, v]) => `${k}=${v}`).join(' · ')}</span>
        )}
        {s.target !== 'sharpness' && before !== undefined && (
          <span className="fact">
            {metricLabel(s.target)} {fmtMetric(s.target, before)} → <b>{fmtMetric(s.target, after)}</b>
          </span>
        )}
        <span className="fact">structure {s.structure}</span>
        <span className="fact dim">{s.ms} ms</span>
      </div>
      {Object.entries(s.revealed || {}).map(([m, [a, b]]) => (
        <span key={m} className="badge revealed">
          revealed hidden {REVEALED_NAMES[m] ?? m}: {fmtMetric(m, a)} → {fmtMetric(m, b)}
        </span>
      ))}
      {!s.kept && <p className="entry-verdict">{s.verdict}</p>}
    </>
  )
}

/** The restorer's log: one entry per agent event, appearing live. Click to view that moment. */
export default function TraceLog({ events, selected, onSelect, running, planner }) {
  const items = events.map((e, i) => ({ ...e, i })).filter((e) => ['start', 'step', 'human', 'question', 'notice', 'error'].includes(e.kind))

  return (
    <aside className="log">
      <header>
        <h2 className="section-title">Restorer’s log</h2>
        {planner && <span className="planner-tag">planner: {planner}</span>}
      </header>
      {items.length === 0 && <p className="log-empty">Each step the agent takes appears here: what it measured, what it chose, and whether the result passed its own check.</p>}
      <ol>
        {items.map((e) => {
          const clickable = Boolean(e.data.image)
          const cls = `entry ${e.kind === 'question' ? 'asked' : e.kind} ${selected === e.i ? 'selected' : ''} ${clickable ? 'clickable' : ''}`
          const body = (() => {
            switch (e.kind) {
              case 'start':
                return <div className="entry-head"><span className="entry-num">0</span><span className="entry-tool">Measured the photo</span></div>
              case 'step':
                return <StepEntry s={e.data} />
              case 'question':
                return (
                  <>
                    <div className="entry-head"><span className="entry-num">?</span><span className="entry-tool">Paused {toolLabel(e.data.about).toLowerCase()}, asking you</span></div>
                    <p className="entry-reason">{e.data.question}</p>
                  </>
                )
              case 'human':
                return (
                  <div className="entry-head">
                    <span className="entry-num">✓</span>
                    <span className="entry-tool">You {e.data.approved ? 'approved' : 'declined'}: {toolLabel(e.data.about).toLowerCase()}</span>
                  </div>
                )
              case 'notice':
                return <p className="entry-reason">{e.data.message}</p>
              case 'error':
                return <p className="entry-reason">Something went wrong: {e.data.message}</p>
              default:
                return null
            }
          })()
          return (
            <li key={e.i} className={cls} onClick={clickable ? () => onSelect(e.i) : undefined}>
              {body}
            </li>
          )
        })}
      </ol>
      {running && <p className="log-working"><span className="dot" /> working…</p>}
    </aside>
  )
}
