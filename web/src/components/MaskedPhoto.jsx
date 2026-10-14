/** A photo with the agent's damage mask (a transparent red PNG) laid over it. */
export default function MaskedPhoto({ src, mask, showMask, alt }) {
  return (
    <div className="masked">
      <img src={src} alt={alt} draggable={false} />
      {mask && <img src={mask} alt="" aria-hidden="true" className={`mask ${showMask ? 'on' : ''}`} draggable={false} />}
    </div>
  )
}
