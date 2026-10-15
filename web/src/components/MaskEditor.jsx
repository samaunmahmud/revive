import { useCallback, useEffect, useRef, useState } from 'react'

// Brush corrections to the agent's damage mask.
//   paint   = damage the detector missed: the agent must repair it
//   protect = real detail it flagged by mistake: the agent must not repaint it
// Strokes are kept as a list (for undo) and replayed into two offscreen layers at
// the photo's own resolution. A later stroke of the other kind cancels an earlier
// one where they overlap, like an eraser. The layers go to the server as PNGs;
// the agent combines them with the mask it measures at repair time.

const COLOURS = { add: 'rgba(255, 209, 102, 0.82)', erase: 'rgba(94, 200, 190, 0.78)' }

function layer(w, h) {
  const c = document.createElement('canvas')
  c.width = w
  c.height = h
  return c
}

function strokePath(ctx, s) {
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.lineWidth = s.r * 2
  ctx.beginPath()
  s.points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)))
  if (s.points.length === 1) ctx.lineTo(s.points[0][0] + 0.01, s.points[0][1])
  ctx.stroke()
}

/** Replays strokes into {add, erase} canvases (opaque white where set). */
function renderLayers(strokes, w, h) {
  const out = { add: layer(w, h), erase: layer(w, h) }
  for (const s of strokes) {
    const mine = out[s.mode].getContext('2d')
    const other = out[s.mode === 'add' ? 'erase' : 'add'].getContext('2d')
    mine.globalCompositeOperation = 'source-over'
    mine.strokeStyle = '#fff'
    strokePath(mine, s)
    other.globalCompositeOperation = 'destination-out'
    other.strokeStyle = '#000'
    strokePath(other, s)
  }
  return out
}

function tinted(src, colour) {
  const c = layer(src.width, src.height)
  const ctx = c.getContext('2d')
  ctx.drawImage(src, 0, 0)
  ctx.globalCompositeOperation = 'source-in'
  ctx.fillStyle = colour
  ctx.fillRect(0, 0, c.width, c.height)
  return c
}

export default function MaskEditor({ image, mask, maskOn = true, kicker, title, intro, onSubmit, onCancel, busy }) {
  const canvas = useRef(null)
  const photo = useRef(null)
  const drawing = useRef(null)
  const frame = useRef(0)
  const [size, setSize] = useState(null)
  const [mode, setMode] = useState('add')
  const [radius, setRadius] = useState(4)
  const [strokes, setStrokes] = useState([])
  const [showAgent, setShowAgent] = useState(maskOn)

  const onLoad = () => {
    const { naturalWidth: w, naturalHeight: h } = photo.current
    if (!w) return
    setSize({ w, h })
    // thin by default: painting just over a mark repairs it best (wider brushes repaint clean pixels too)
    setRadius(Math.max(2, Math.round(w / 300)))
  }

  // a cached photo can finish loading before React attaches onLoad
  useEffect(() => { if (photo.current?.complete) onLoad() }, [image]) // eslint-disable-line react-hooks/exhaustive-deps

  const redraw = useCallback((list) => {
    const c = canvas.current
    if (!c || !size) return
    const ctx = c.getContext('2d')
    ctx.clearRect(0, 0, c.width, c.height)
    const layers = renderLayers(list, size.w, size.h)
    ctx.drawImage(tinted(layers.add, COLOURS.add), 0, 0)
    ctx.drawImage(tinted(layers.erase, COLOURS.erase), 0, 0)
  }, [size])

  useEffect(() => { redraw(strokes) }, [strokes, redraw])

  const undo = useCallback(() => setStrokes((s) => s.slice(0, -1)), [])
  useEffect(() => {
    const key = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undo() }
    }
    window.addEventListener('keydown', key)
    return () => window.removeEventListener('keydown', key)
  }, [undo])

  const point = (e) => {
    const r = canvas.current.getBoundingClientRect()
    return [((e.clientX - r.left) / r.width) * size.w, ((e.clientY - r.top) / r.height) * size.h]
  }

  const down = (e) => {
    if (!size || busy) return
    e.currentTarget.setPointerCapture(e.pointerId)
    drawing.current = { mode, r: radius, points: [point(e)] }
    redraw([...strokes, drawing.current])
  }
  const move = (e) => {
    if (!drawing.current) return
    drawing.current.points.push(point(e))
    cancelAnimationFrame(frame.current)
    frame.current = requestAnimationFrame(() => drawing.current && redraw([...strokes, drawing.current]))
  }
  const up = () => {
    if (!drawing.current) return
    const s = drawing.current
    drawing.current = null
    setStrokes((list) => [...list, s])
  }

  const counts = { add: strokes.filter((s) => s.mode === 'add').length, erase: strokes.filter((s) => s.mode === 'erase').length }

  const submit = () => {
    const layers = renderLayers(strokes, size.w, size.h)
    onSubmit({
      add: counts.add ? layers.add.toDataURL('image/png') : null,
      erase: counts.erase ? layers.erase.toDataURL('image/png') : null,
    })
  }

  // brush preview size on screen (the canvas is scaled to fit)
  const shownWidth = canvas.current?.getBoundingClientRect().width
  const brushPx = size && shownWidth ? Math.max(4, (radius * 2 * shownWidth) / size.w) : radius * 2

  return (
    <section className="mask-editor" aria-label="Correct the damage mask">
      <div className="mask-editor-head">
        <span className="question-kicker">{kicker}</span>
        <h3>{title}</h3>
        <p>{intro}</p>
      </div>

      <div className="mask-tools" role="toolbar" aria-label="Mask brush">
        <div className="segmented" role="radiogroup" aria-label="Brush">
          <button type="button" role="radio" aria-checked={mode === 'add'} className={mode === 'add' ? 'on paint' : ''} onClick={() => setMode('add')}>
            <span className="swatch paint" /> Paint damage
          </button>
          <button type="button" role="radio" aria-checked={mode === 'erase'} className={mode === 'erase' ? 'on protect' : ''} onClick={() => setMode('erase')}>
            <span className="swatch protect" /> Protect detail
          </button>
        </div>
        <label className="brush-size">
          <span>Brush</span>
          <input type="range" min="1" max={size ? Math.max(8, Math.round(size.w / 40)) : 40} value={radius}
            onChange={(e) => setRadius(Number(e.target.value))} aria-label="Brush size" />
          <span className="brush-dot" style={{ width: brushPx, height: brushPx }} aria-hidden="true" />
        </label>
        <button type="button" className="btn ghost small" onClick={undo} disabled={!strokes.length}>Undo</button>
        <button type="button" className="btn ghost small" onClick={() => setStrokes([])} disabled={!strokes.length}>Clear</button>
        {mask && (
          <label className="toggle"><input type="checkbox" checked={showAgent} onChange={(e) => setShowAgent(e.target.checked)} /> agent’s mask</label>
        )}
      </div>

      <div className="print large">
        <div className={`mask-canvas ${mode}`}>
          <img ref={photo} src={image} alt="Photo to mark up" onLoad={onLoad} draggable={false} />
          {mask && <img src={mask} alt="" aria-hidden="true" className={`mask ${showAgent ? 'on' : ''}`} draggable={false} />}
          {size && (
            <canvas
              ref={canvas}
              width={size.w}
              height={size.h}
              onPointerDown={down}
              onPointerMove={move}
              onPointerUp={up}
              onPointerCancel={up}
              aria-label="Drawing surface: paint over marks to repair, or over detail to protect"
            />
          )}
        </div>
      </div>

      <div className="mask-legend">
        <span><i className="swatch damage" /> found by OpenCV</span>
        <span><i className="swatch paint" /> you: repair this</span>
        <span><i className="swatch protect" /> you: leave this alone</span>
      </div>

      <div className="question-actions">
        <button type="button" className="btn primary" onClick={submit} disabled={busy || !size || !strokes.length}>
          Repair with my corrections
        </button>
        <button type="button" className="btn" onClick={onCancel} disabled={busy}>Cancel</button>
        <span className="mask-hint">
          {strokes.length
            ? `${counts.add} paint · ${counts.erase} protect stroke${counts.add + counts.erase === 1 ? '' : 's'}. The agent re-runs, repairs what you painted after fixing the tones, and checks each painted mark now blends in.`
            : 'Paint just over each mark; a thin brush gives the cleanest repair.'}
        </span>
      </div>
    </section>
  )
}
