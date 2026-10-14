import { fmtMetric, toolLabel } from '../labels'
import MaskedPhoto from './MaskedPhoto'

const COPY = {
  inpaint_scratches: {
    yes: 'Approve repair mask',
    no: 'Leave the damage',
    explain: (q, ev) =>
      `Red marks what OpenCV detected as scratches and dust (${((ev?.mask_coverage ?? q.measurements.scratch_coverage) * 100).toFixed(1)}% of the photo). Filling that much means inventing pixels, so the agent wants a person to check the mask first.`,
  },
  correct_colour: {
    yes: 'It’s ageing, remove it',
    no: 'It’s the real colour',
    explain: (q) =>
      `The highlights are tinted (cast ${fmtMetric('highlight_cast', q.measurements.highlight_cast)}), but the blacks are intact (black point ${fmtMetric('black_point', q.measurements.black_point)}). Aged prints usually fade both, so this may be the scene’s true colour.`,
  },
}

/** A question the agent paused on, with the OpenCV evidence and approve/decline buttons. */
export default function QuestionCard({ question, evidence, onAnswer, busy }) {
  const copy = COPY[question.about]
  const showsMask = question.about === 'inpaint_scratches' && evidence?.mask

  if (!copy) {
    // residual_damage: informational. The result is ready, but a person should look at it.
    return (
      <section className="question note">
        <h3>Worth a human look</h3>
        <p>{question.question} Automatic repair has done what it safely can; what remains needs a person’s judgement.</p>
      </section>
    )
  }

  return (
    <section className="question">
      <div className="question-text">
        <span className="question-kicker">The agent paused · {toolLabel(question.about)}</span>
        <h3>{question.question}</h3>
        <p>{copy.explain(question, evidence)}</p>
        <div className="question-actions">
          <button type="button" className="btn primary" disabled={busy} onClick={() => onAnswer(question.about, true)}>{copy.yes}</button>
          <button type="button" className="btn" disabled={busy} onClick={() => onAnswer(question.about, false)}>{copy.no}</button>
        </div>
      </div>
      {evidence?.image && (
        <figure className="question-evidence">
          <span className="print small"><MaskedPhoto src={evidence.image} mask={showsMask ? evidence.mask : null} showMask alt="Evidence for the question" /></span>
          <figcaption>{showsMask ? 'Detected damage in red' : 'The photo when the agent asked'}</figcaption>
        </figure>
      )}
    </section>
  )
}
