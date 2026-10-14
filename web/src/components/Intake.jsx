import { useRef, useState } from 'react'

/** Empty state: drop a photo, or pick one of the bundled damaged samples. */
export default function Intake({ samples, onFile, onSample, busy }) {
  const input = useRef(null)
  const [over, setOver] = useState(false)

  const drop = (e) => {
    e.preventDefault()
    setOver(false)
    const f = e.dataTransfer.files?.[0]
    if (f) onFile(f)
  }

  return (
    <div className="intake">
      <button
        type="button"
        className={`dropzone ${over ? 'over' : ''}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true) }}
        onDragLeave={() => setOver(false)}
        onDrop={drop}
        disabled={busy}
      >
        <span className="dropzone-title">Drop an old photo here</span>
        <span className="dropzone-sub">or click to choose a file · JPEG, PNG, WebP, TIFF · up to 15 MB</span>
        <input
          ref={input}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
        />
      </button>

      <div className="contact-sheet">
        <h2 className="section-title">Or try a damaged print from the archive</h2>
        <ul className="samples">
          {samples.map((s) => (
            <li key={s.id}>
              <button type="button" className="sample" onClick={() => onSample(s.id)} disabled={busy}>
                <span className="print"><img src={s.url} alt={s.title} loading="lazy" /></span>
                <span className="sample-title">{s.title}</span>
                <span className="sample-desc">{s.description}</span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
