import { useEffect, useMemo, useState } from 'react'
import { answerJob, fullResJob, getConfig, getSamples, startJob } from './api'
import { useJobStream } from './useJobStream'
import { fmtScore, toolLabel } from './labels'
import { summarise } from './summary'
import Intake from './components/Intake'
import CompareSlider from './components/CompareSlider'
import MaskedPhoto from './components/MaskedPhoto'
import Diagnosis from './components/Diagnosis'
import TraceLog from './components/TraceLog'
import QuestionCard from './components/QuestionCard'
import AgentLoop from './components/AgentLoop'

export default function App() {
  const [config, setConfig] = useState(null)
  const [samples, setSamples] = useState([])
  const [planner, setPlanner] = useState('heuristic')
  const [jobId, setJobId] = useState(null)
  const [answers, setAnswers] = useState({})
  const [selected, setSelected] = useState(null)   // index into events, or null = default view
  const [showMask, setShowMask] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    getConfig().then(setConfig).catch(() => {})
    getSamples().then(setSamples).catch((e) => setError(e.message))
  }, [])

  const { events } = useJobStream(jobId)

  // Everything on screen is derived from the event list.
  const view = useMemo(() => {
    const start = events.find((e) => e.kind === 'start')?.data
    const done = events.find((e) => e.kind === 'done')?.data
    const failed = events.find((e) => e.kind === 'error')?.data
    const latestWithImage = [...events].reverse().find((e) => e.data.image)
    const questionEvents = events.filter((e) => e.kind === 'question').map((e) => e.data)
    const latest = [...events].reverse().find((e) => ['start', 'step', 'question', 'human'].includes(e.kind))
    return { start, done, failed, latest, latestWithImage, questionEvents, running: Boolean(jobId) && !done && !failed }
  }, [events, jobId])

  const run = async (fn) => {
    setBusy(true)
    setError('')
    try {
      const { job_id } = await fn()
      setSelected(null)
      setShowMask(true)
      setJobId(job_id)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const newPhoto = (input) => { setAnswers({}); run(() => startJob({ ...input, planner })) }
  const answer = (about, approved) => {
    setAnswers((a) => ({ ...a, [about]: approved }))
    run(() => answerJob(jobId, { [about]: approved }))
  }
  const fullSize = () => run(() => fullResJob(jobId))
  const reset = () => { setJobId(null); setAnswers({}); setSelected(null); setError('') }

  const trace = view.done?.trace
  const selectedEvent = selected !== null ? events[selected] : null
  const shown = selectedEvent ?? (view.done ? null : view.latestWithImage)
  const bedrock = config?.bedrock

  return (
    <div className="app">
      <header className="masthead">
        <div className="brand">
          <h1>ReVive</h1>
          <p>An agent that restores old photos the way a restorer would: it measures the damage with OpenCV 5, repairs one thing at a time, checks every repair, and asks you when it isn’t sure.</p>
        </div>
        <div className="planner" role="radiogroup" aria-label="Planner">
          <span className="planner-label">Planner</span>
          <button type="button" role="radio" aria-checked={planner === 'heuristic'} className={planner === 'heuristic' ? 'on' : ''} onClick={() => setPlanner('heuristic')}>Heuristic</button>
          <button
            type="button"
            role="radio"
            aria-checked={planner === 'bedrock'}
            className={planner === 'bedrock' ? 'on' : ''}
            onClick={() => setPlanner('bedrock')}
            disabled={!bedrock?.available}
            title={bedrock?.available ? `${bedrock.model_id} · ${bedrock.region}` : 'No AWS credentials on this server'}
          >
            Bedrock LLM
          </button>
        </div>
      </header>

      {error && <div className="alert" role="alert">{error}</div>}

      {!jobId ? (
        <Intake samples={samples} busy={busy} onFile={(file) => newPhoto({ file })} onSample={(sample) => newPhoto({ sample })} />
      ) : (
        <main className="workbench">
          <div className="stage">
            <AgentLoop event={view.latest} done={Boolean(view.done)} eventKey={events.length} />
            <div className="lighttable">
              <div className="viewer-bar">
                <span className="viewer-caption">
                  {selectedEvent
                    ? captionFor(selectedEvent)
                    : view.done ? 'Before / after: drag to compare'
                    : view.running ? 'Restoring…' : ''}
                </span>
                <span className="viewer-tools">
                  {shown?.data.mask && (
                    <label className="toggle"><input type="checkbox" checked={showMask} onChange={(e) => setShowMask(e.target.checked)} /> damage mask</label>
                  )}
                  {selectedEvent && view.done && <button type="button" className="link" onClick={() => setSelected(null)}>back to before / after</button>}
                </span>
              </div>
              <div className="print large">
                {shown ? (
                  <MaskedPhoto src={shown.data.image} mask={shown.data.mask} showMask={showMask} alt={captionFor(shown)} />
                ) : view.done ? (
                  <CompareSlider before={view.done.original} after={view.done.restored} />
                ) : (
                  <div className="developing">developing…</div>
                )}
              </div>
            </div>

            {view.failed && <div className="alert">The restoration failed: {view.failed.message}</div>}

            {trace && (
              <>
                <section className={`outcome ${trace.status}`}>
                  <div className="outcome-score" title="Damage score (lower is better)">
                    <span>{fmtScore(trace.initial_score)}</span>
                    <span className="arrow">→</span>
                    <b>{fmtScore(trace.final_score)}</b>
                  </div>
                  <div className="outcome-text">
                    <span className="status-pill">{trace.status === 'complete' ? 'Restored' : 'Needs your review'}</span>
                    <span className="outcome-meta">
                      {trace.size && <>{trace.size[0]}×{trace.size[1]}{trace.full_res ? ' (full size)' : view.done.full_size ? ' preview' : ''} · </>}
                      {trace.steps.filter((s) => s.kept).length} repairs kept · {trace.steps.filter((s) => !s.kept).length} rejected · {trace.total_ms} ms · planner {trace.planner}
                    </span>
                    {view.done.archive && <span className="outcome-meta">archived to {view.done.archive}</span>}
                    {Object.keys(answers).length > 0 && (
                      <span className="outcome-meta">your answers: {Object.entries(answers).map(([k, v]) => `${toolLabel(k).toLowerCase()} ${v ? '✓' : '✗'}`).join(' · ')}</span>
                    )}
                  </div>
                  <div className="outcome-actions">
                    {view.done.full_size && !trace.full_res && (
                      <button type="button" className="btn primary" onClick={fullSize} disabled={busy}
                        title="Runs the agent again on your full-size upload with the same answers">
                        Restore at full size ({view.done.full_size[0]}×{view.done.full_size[1]})
                      </button>
                    )}
                    <a className={`btn ${view.done.full_size && !trace.full_res ? '' : 'primary'}`} href={`${view.done.restored_png}?download=true`}>Download photo</a>
                    <a className="btn" href={`/api/jobs/${jobId}/files/trace.json?download=true`}>Download trace</a>
                    <button type="button" className="btn ghost" onClick={reset}>New photo</button>
                  </div>
                </section>

                <section className="summary">
                  <h2 className="section-title">What ReVive did</h2>
                  <ul>{summarise(trace).map((line) => <li key={line}>{line}</li>)}</ul>
                </section>

                {trace.questions.map((q) => (
                  <QuestionCard
                    key={`${jobId}-${q.about}`}
                    question={q}
                    evidence={view.questionEvents.find((e) => e.about === q.about)}
                    onAnswer={answer}
                    busy={busy}
                  />
                ))}
              </>
            )}

            <Diagnosis initial={view.start?.initial} initialScore={view.start?.initial_score} final={trace?.final} finalScore={trace?.final_score} />
          </div>

          <TraceLog
            events={events}
            selected={selected}
            onSelect={(i) => setSelected(i === selected ? null : i)}
            running={view.running}
            planner={view.start?.planner}
          />
        </main>
      )}

      <footer className="colophon">
        OpenCV 5 perception and repair · plan → act → verify loop · human approval for risky repairs · every step logged
      </footer>
    </div>
  )
}

function captionFor(e) {
  switch (e.kind) {
    case 'start': return 'As uploaded, with the damage OpenCV found'
    case 'step': return `Step ${e.data.step}: ${toolLabel(e.data.tool)} (${e.data.kept ? 'kept' : 'rejected, not applied'})`
    case 'question': return `Paused to ask about: ${toolLabel(e.data.about).toLowerCase()}`
    default: return ''
  }
}
