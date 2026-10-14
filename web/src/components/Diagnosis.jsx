import { METRICS, fmtMetric, fmtScore } from '../labels'

/** The OpenCV measurements at the start, and (once finished) at the end. */
export default function Diagnosis({ initial, final, initialScore, finalScore }) {
  if (!initial) return null
  return (
    <section className="diagnosis">
      <header>
        <h2 className="section-title">Diagnosis · OpenCV 5</h2>
        <span className="score">
          damage score <b>{fmtScore(initialScore)}</b>
          {finalScore !== undefined && <> → <b className={finalScore <= 1 ? 'good' : 'warn'}>{fmtScore(finalScore)}</b></>}
        </span>
      </header>
      <dl>
        {Object.entries(METRICS).map(([key, m]) => {
          const a = initial[key]
          const b = final?.[key]
          return (
            <div key={key} className={`metric ${m.bad(a) ? 'bad' : ''}`} title={m.hint}>
              <dt>{m.label}</dt>
              <dd>
                {fmtMetric(key, a)}
                {b !== undefined && b !== a && <span className={`after ${m.bad(b) ? 'bad' : ''}`}> → {fmtMetric(key, b)}</span>}
              </dd>
            </div>
          )
        })}
      </dl>
    </section>
  )
}
