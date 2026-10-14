import { useRef, useState } from 'react'

const HOW = [
  { n: '01', title: 'Perceive', text: 'OpenCV 5 measures scratches, grain, faded blacks, colour cast, contrast and sharpness.' },
  { n: '02', title: 'Plan', text: 'A planner picks one repair from those numbers. It never sees the pixels.' },
  { n: '03', title: 'Act', text: 'OpenCV 5 applies it: colour, tones, inpainting or denoising.' },
  { n: '04', title: 'Verify', text: 'Re-measure. Keep the repair only if it helped and the structure survived; otherwise retry gentler.' },
  { n: '05', title: 'Ask', text: 'If a repair could change what the photo really shows, it asks you instead of guessing.' },
]

/** Empty state: what the agent does, then drop a photo or pick a bundled damaged sample. */
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
      <ol className="how">
        {HOW.map((h) => (
          <li key={h.n}>
            <span className="how-n">{h.n}</span>
            <span className="how-title">{h.title}</span>
            <span className="how-text">{h.text}</span>
          </li>
        ))}
      </ol>

      <div className="contact-sheet">
        <div className="sheet-head">
          <h2 className="section-title">Try a damaged print from the archive</h2>
          <span className="sheet-note">one click · runs the agent live</span>
        </div>
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

      <button
        type="button"
        className={`dropzone ${over ? 'over' : ''} ${busy ? 'busy' : ''}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setOver(true) }}
        onDragLeave={() => setOver(false)}
        onDrop={drop}
        disabled={busy}
      >
        <svg className="dropzone-icon" viewBox="0 0 48 48" aria-hidden="true">
          <rect x="7" y="9" width="34" height="30" rx="2" fill="none" stroke="currentColor" strokeWidth="2" />
          <path d="M12 33l9-10 6 6 4-4 6 8" fill="none" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" />
          <circle cx="32" cy="17" r="3" fill="currentColor" />
        </svg>
        <span className="dropzone-title">{busy ? 'Uploading…' : 'Or restore your own old photo'}</span>
        <span className="dropzone-sub">Drop it here or click to choose · JPEG, PNG, WebP, TIFF · up to 15 MB</span>
        <input
          ref={input}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
        />
      </button>
    </div>
  )
}
