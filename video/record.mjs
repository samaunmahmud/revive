// Record demo scenes of the running ReVive app with headless Chrome (DevTools protocol).
//
//   node video/record.mjs [baseUrl] [outDir]      # defaults: http://localhost:8000  video/out/clips
//
// Needs Chrome started with --remote-debugging-port=9333 (make_video.py does this).
// Each scene is saved as JPEG frames plus their timestamps, so make_video.py can
// rebuild real-time footage at a constant frame rate. A drawn cursor glides to
// every button before it is clicked, so viewers can follow the interaction.
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

const BASE = process.argv[2] || 'http://localhost:8000'
const OUT = process.argv[3] || 'video/out/clips'
const W = 1280, H = 720, SCALE = 1.5          // 1920x1080 frames; UI drawn 1.5x so it's legible on video

const CURSOR = `
(() => {
  if (window.__cursor) return;
  const make = () => {
    const c = document.createElement('div');
    c.id = '__cursor';
    c.innerHTML = '<svg width="26" height="26" viewBox="0 0 26 26"><path d="M3 2 L3 21 L8 16 L12 24 L15 22.5 L11 15 L18 15 Z" fill="#fff" stroke="#111" stroke-width="1.6" stroke-linejoin="round"/></svg><span></span>';
    Object.assign(c.style, { position: 'fixed', left: '0', top: '0', zIndex: 2147483647, pointerEvents: 'none',
      transform: 'translate(640px, 420px)', transition: 'transform 0.8s cubic-bezier(.4,0,.2,1)', filter: 'drop-shadow(0 2px 3px rgba(0,0,0,.5))' });
    const ring = c.querySelector('span');
    Object.assign(ring.style, { position: 'absolute', left: '-14px', top: '-14px', width: '30px', height: '30px', borderRadius: '50%',
      border: '3px solid #e2a04f', opacity: '0', transform: 'scale(.4)', transition: 'all .45s ease-out' });
    document.body.appendChild(c);
    window.__cursor = c;
    window.__moveCursor = (x, y, ms) => { c.style.transitionDuration = ms + 'ms'; c.style.transform = 'translate(' + x + 'px,' + y + 'px)'; };
    window.__pulse = () => { ring.style.transition = 'none'; ring.style.opacity = '1'; ring.style.transform = 'scale(.4)';
      requestAnimationFrame(() => { ring.style.transition = 'all .45s ease-out'; ring.style.opacity = '0'; ring.style.transform = 'scale(1.6)'; }); };
  };
  if (document.body) make(); else addEventListener('DOMContentLoaded', make);
})();`

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

async function connect() {
  const t = await (await fetch(`http://127.0.0.1:9333/json/new?about:blank`, { method: 'PUT' })).json()
  const ws = new WebSocket(t.webSocketDebuggerUrl)
  let id = 0
  const pending = {}, listeners = []
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data)
    if (d.id && pending[d.id]) { pending[d.id](d); delete pending[d.id] }
    else if (d.method) listeners.forEach((f) => f(d))
  }
  await new Promise((r) => (ws.onopen = r))
  const send = (method, params = {}) => new Promise((r) => { pending[++id] = r; ws.send(JSON.stringify({ id, method, params })) })
  return { ws, send, on: (f) => listeners.push(f) }
}

async function main() {
  const { send, on } = await connect()
  const js = async (expr) => (await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value
  await send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: SCALE, mobile: false })
  await send('Page.enable')
  await send('Page.addScriptToEvaluateOnNewDocument', { source: CURSOR })

  // ---- recording ----
  let frames = null
  on(async (msg) => {
    if (msg.method !== 'Page.screencastFrame') return
    const { data, metadata, sessionId } = msg.params
    send('Page.screencastFrameAck', { sessionId })
    if (frames) frames.push({ t: metadata.timestamp, data })
  })
  async function record(name, actions) {
    frames = []
    await send('Page.startScreencast', { format: 'jpeg', quality: 88, maxWidth: W * SCALE, maxHeight: H * SCALE, everyNthFrame: 1 })
    const t0 = Date.now() / 1000
    await actions()
    await sleep(300)
    await send('Page.stopScreencast')
    const dir = join(OUT, name)
    mkdirSync(dir, { recursive: true })
    const index = frames.map((f, i) => {
      writeFileSync(join(dir, `${String(i).padStart(5, '0')}.jpg`), Buffer.from(f.data, 'base64'))
      return { file: `${String(i).padStart(5, '0')}.jpg`, t: f.t }
    })
    writeFileSync(join(dir, 'frames.json'), JSON.stringify({ start: t0, end: Date.now() / 1000, frames: index }))
    console.log(`${name}: ${index.length} frames, ${(Date.now() / 1000 - t0).toFixed(1)} s`)
    frames = null
  }

  // ---- interaction helpers (cursor glides, then clicks) ----
  async function home() {
    await send('Page.navigate', { url: BASE })
    await sleep(1800)
    await js('window.scrollTo(0, 0)')
  }
  async function center(selector, text) {
    return js(`(() => { const els = [...document.querySelectorAll(${JSON.stringify(selector)})];
      const el = ${text ? `els.find(e => e.textContent.includes(${JSON.stringify(text)}))` : 'els[0]'};
      if (!el) return null; el.scrollIntoView({block: 'center', behavior: 'smooth'}); return true })()`)
  }
  async function click(selector, text, move = 900) {
    if (!(await center(selector, text))) throw new Error(`not found: ${selector} ${text || ''}`)
    await sleep(700)
    const p = await js(`(() => { const els = [...document.querySelectorAll(${JSON.stringify(selector)})];
      const el = ${text ? `els.find(e => e.textContent.includes(${JSON.stringify(text)}))` : 'els[0]'};
      const r = el.getBoundingClientRect(); return [r.left + r.width / 2, r.top + Math.min(r.height / 2, 60)] })()`)
    await js(`window.__moveCursor(${p[0]}, ${p[1]}, ${move})`)
    await sleep(move + 250)
    await js('window.__pulse()')
    await sleep(180)
    await js(`(() => { const els = [...document.querySelectorAll(${JSON.stringify(selector)})];
      const el = ${text ? `els.find(e => e.textContent.includes(${JSON.stringify(text)}))` : 'els[0]'}; el.click() })()`)
  }
  async function waitDone(timeout = 60000) {
    const t = Date.now()
    while (Date.now() - t < timeout) {
      if (await js(`!!document.querySelector('.outcome')`)) return
      await sleep(250)
    }
    throw new Error('restoration did not finish')
  }
  async function waitRerun(timeout = 60000) {
    // after answering a question the old result is still on screen for a moment:
    // wait for it to go, then for the new run to finish
    const t = Date.now()
    while (Date.now() - t < timeout && await js(`!!document.querySelector('.outcome')`)) await sleep(100)
    await waitDone(timeout)
  }
  async function scrollTo(selector, block = 'center') {
    await js(`document.querySelector(${JSON.stringify(selector)})?.scrollIntoView({block: '${block}', behavior: 'smooth'})`)
    await sleep(1100)
  }
  async function sweepSlider() {
    // drag the before/after handle: 50 -> 12 -> 88 -> 50, with the cursor following it
    const box = await js(`(() => { const r = document.querySelector('.compare').getBoundingClientRect(); return [r.left, r.top, r.width, r.height] })()`)
    if (!box) return
    const [x, y, w, h] = box
    const set = (v) => js(`(() => { const el = document.querySelector('.compare-range');
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(el, ${v});
      el.dispatchEvent(new Event('input', { bubbles: true })); })()`)
    await js(`window.__moveCursor(${x + w / 2}, ${y + h / 2}, 700)`)
    await sleep(900)
    for (const [from, to, ms] of [[50, 12, 1600], [12, 88, 2600], [88, 50, 1400]]) {
      const steps = Math.round(ms / 33)
      for (let i = 1; i <= steps; i++) {
        const e = i / steps, v = from + (to - from) * (e < 0.5 ? 2 * e * e : 1 - Math.pow(-2 * e + 2, 2) / 2)
        await set(v.toFixed(1))
        await js(`window.__moveCursor(${x + (w * v) / 100}, ${y + h / 2}, 0)`)
        await sleep(33)
      }
      await sleep(500)
    }
  }

  // ---- scenes ----
  await record('01-landing', async () => {
    await home()
    await sleep(2200)
    await js(`window.__moveCursor(560, 300, 1400)`); await sleep(1600)
    await js(`document.querySelector('.contact-sheet').scrollIntoView({block: 'start', behavior: 'smooth'})`)
    await sleep(2600)
  })
  await record('02-hero', async () => {
    await home()
    await click('.sample', 'Faded portrait')
    await waitDone()
    await sleep(1800)
    await sweepSlider()
    await scrollTo('.summary')
    await sleep(4200)
  })
  await record('03-mask-approval', async () => {
    await home()
    await click('.sample', 'Badly damaged cat')
    await waitDone()
    await sleep(900)
    await scrollTo('.question:not(.note)')
    await sleep(3800)
    await click('.question .btn.primary', 'Approve')
    await waitRerun()
    await sleep(1600)
    await scrollTo('.compare')
    await sweepSlider()
    await sleep(800)
  })
  await record('04-real-slide', async () => {
    await home()
    await click('.sample', 'Real 1972 colour slide')
    await waitDone()
    await sleep(800)
    await scrollTo('.question:not(.note)')
    await sleep(3800)
    await click('.question .btn.primary', 'ageing')
    await waitRerun()
    await sleep(1400)
    await scrollTo('.compare')
    await sweepSlider()
    await sleep(800)
  })
  await record('05-control', async () => {
    await home()
    await click('.sample', 'Undamaged control')
    await waitDone()
    await sleep(1500)
    await scrollTo('.summary')
    await sleep(3500)
  })
  await record('06-real-portrait', async () => {
    await home()
    await click('.sample', 'Real 1925 portrait')
    await waitDone()
    await sleep(1200)
    await sweepSlider()
    await sleep(600)
  })
  await record('07-touch-up', async () => {
    // the agent misses a wide real scratch; a person paints it and the agent repairs and checks it
    await home()
    await click('.sample', 'Real 1942')
    await waitDone()
    await sleep(1200)
    await scrollTo('.outcome')
    await sleep(1500)
    await click('.outcome .btn', 'Touch up')
    await sleep(600)
    await scrollTo('.mask-editor', 'start')
    await sleep(1200)
    // the scratch, as fractions of the photo (traced by eval/human_edits.py's band method)
    const SCRATCH = [[0.6963, 0.1709], [0.6987, 0.1915], [0.7037, 0.212], [0.7075, 0.2325], [0.71, 0.253],
      [0.7113, 0.2735], [0.7163, 0.294], [0.7188, 0.3145], [0.72, 0.335], [0.7225, 0.3556], [0.7262, 0.3761],
      [0.7262, 0.3966], [0.73, 0.4171], [0.7312, 0.4376], [0.73, 0.4581], [0.7362, 0.4786], [0.7362, 0.4991],
      [0.735, 0.5197], [0.7262, 0.5402], [0.7262, 0.5607], [0.725, 0.5812], [0.73, 0.6017], [0.7288, 0.6222], [0.7325, 0.6427]]
    const box = await js(`(() => { const r = document.querySelector('.mask-canvas canvas').getBoundingClientRect(); return [r.left, r.top, r.width, r.height] })()`)
    const pts = SCRATCH.map(([u, v]) => [box[0] + u * box[2], box[1] + v * box[3]])
    await js(`window.__moveCursor(${pts[0][0]}, ${pts[0][1]}, 900)`)
    await sleep(1150)
    await send('Input.dispatchMouseEvent', { type: 'mousePressed', x: pts[0][0], y: pts[0][1], button: 'left', buttons: 1, clickCount: 1 })
    for (let i = 1; i < pts.length; i++) {   // a slow, hand-speed stroke with the cursor on it
      for (let k = 1; k <= 4; k++) {
        const x = pts[i - 1][0] + (pts[i][0] - pts[i - 1][0]) * k / 4, y = pts[i - 1][1] + (pts[i][1] - pts[i - 1][1]) * k / 4
        await send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y, button: 'left', buttons: 1 })
        await js(`window.__moveCursor(${x}, ${y}, 0)`)
        await sleep(28)
      }
    }
    const last = pts[pts.length - 1]
    await send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: last[0], y: last[1], button: 'left', buttons: 0, clickCount: 1 })
    await sleep(1600)
    await click('.mask-editor .btn.primary', 'Repair with my corrections')
    await waitRerun()
    await sleep(1200)
    await scrollTo('.compare')
    await sweepSlider()
    await scrollTo('.summary')
    await sleep(3500)
  })
  await send('Page.close').catch(() => {})
  process.exit(0)
}

main().catch((e) => { console.error(e); process.exit(1) })
