import { useRef, useState } from 'react'

/** Before/after slider: the restored photo underneath, the original clipped on top. */
export default function CompareSlider({ before, after }) {
  const [pos, setPos] = useState(50)
  const box = useRef(null)
  const dragging = useRef(false)

  const moveTo = (clientX) => {
    const r = box.current.getBoundingClientRect()
    setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)))
  }

  return (
    <div
      ref={box}
      className="compare"
      onPointerDown={(e) => { dragging.current = true; e.currentTarget.setPointerCapture(e.pointerId); moveTo(e.clientX) }}
      onPointerMove={(e) => dragging.current && moveTo(e.clientX)}
      onPointerUp={() => { dragging.current = false }}
    >
      <img src={after} alt="Restored photo" draggable={false} />
      <img
        src={before}
        alt="Original photo"
        draggable={false}
        className="compare-before"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
      />
      <div className="compare-handle" style={{ left: `${pos}%` }} aria-hidden="true"><span /></div>
      <span className="compare-tag left">Before</span>
      <span className="compare-tag right">After</span>
      <input
        type="range"
        min="0"
        max="100"
        value={pos}
        onChange={(e) => setPos(Number(e.target.value))}
        aria-label="Compare before and after"
        className="compare-range"
      />
    </div>
  )
}
