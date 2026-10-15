"""Assemble the captioned demo-video draft from recorded app scenes and slides.

    node video/record.mjs                 # 1. record the app (needs the app on :8000 and Chrome on :9333)
    python -m video.make_video            # 2. -> video/out/ReVive-draft.mp4 (1920x1080, H.264, 30 fps)

Slides and captions are rendered as HTML by headless Chrome (same fonts and
colours as the app), app scenes are re-timed from their screencast timestamps,
and segments are joined with short crossfades. Captions follow docs/VIDEO_SCRIPT.md
in short form; record a voice-over on top, and replace the team placeholder with
a short clip of yourself.
"""
import glob
import html
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIPS = os.path.join(ROOT, "video", "out", "clips")
OUT = os.path.join(ROOT, "video", "out", "ReVive-draft.mp4")
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
W, H, FPS, FADE = 1920, 1080, 30, 0.4
FIG_H = 700   # tallest figure that still ends above a two-line caption (captions start near y=900)
LIVE_URL = os.environ.get("REVIVE_URL", "live demo link: add after deployment")
FONTS = os.path.join(ROOT, "web", "node_modules", "@fontsource-variable")

CSS = f"""
@import url("file://{FONTS}/fraunces/full.css");
@import url("file://{FONTS}/fraunces/full-italic.css");
@import url("file://{FONTS}/inter/index.css");
* {{ box-sizing: border-box; margin: 0; }}
body {{ width: {W}px; height: {H}px; overflow: hidden; font-family: 'Inter Variable', sans-serif; color: #eadfca;
  background: radial-gradient(ellipse at 30% -10%, rgba(226,160,79,.14), transparent 55%), #15110e; }}
.serif {{ font-family: 'Fraunces Variable', Georgia, serif; }}
"""


def render_html(body, path, transparent=False, extra_css=""):
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as fh:
        fh.write(f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}{extra_css}</style></head><body>{body}</body></html>")
        src = fh.name
    for attempt in range(3):   # each render gets its own throwaway profile; retry transient Chrome failures
        profile = tempfile.mkdtemp(prefix="revive-chrome-")
        args = [CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={profile}",
                f"--window-size={W},{H}", "--virtual-time-budget=2500", f"--screenshot={path}", f"file://{src}"]
        if transparent:
            args.insert(4, "--default-background-color=00000000")
        done = subprocess.run(args, capture_output=True).returncode == 0 and os.path.exists(path)
        shutil.rmtree(profile, ignore_errors=True)
        if done:
            break
    else:
        sys.exit(f"Chrome could not render {path}")
    os.unlink(src)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img if img.shape[:2] == (H, W) else cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)


def slide(tmp, name, kicker, title, body="", image=None, image_h=FIG_H, centre=False):
    """Centred slide: kicker, big serif title, then either body text or a figure."""
    if image:
        # explicit height: CSS max-* never scales a small figure UP to fill the slide
        content = (f"<img src='file://{image}' style='height:{image_h}px;width:auto;max-width:1820px;object-fit:contain;"
                   "display:block;margin:22px auto 0;border-radius:12px;background:#fcfbf8;box-shadow:0 12px 40px rgba(0,0,0,.5)'>")
        title_size, pad, align = 50, "44px 50px 0", "center" if centre else "flex-start"   # top-aligned so the caption never covers the figure
    else:
        content = f"<div style='font-size:42px;line-height:1.55;color:#d6c9b1;margin:34px auto 0;max-width:1500px'>{body}</div>"
        title_size, pad, align = 104, "0 120px", "center"
    return render_html(f"""
      <div style="height:{H}px;display:flex;flex-direction:column;justify-content:{align};text-align:center;padding:{pad}">
        <div style="font:700 26px Inter Variable;letter-spacing:.18em;text-transform:uppercase;color:#e2a04f">{kicker}</div>
        <div class="serif" style="font-style:italic;font-weight:600;font-size:{title_size}px;line-height:1.1;margin-top:14px;color:#efe5d2">{title}</div>
        {content}
      </div>""", os.path.join(tmp, f"{name}.png"))[:, :, :3]


def caption(tmp, i, text):
    return render_html(f"""<div style="position:absolute;left:0;right:0;bottom:46px;text-align:center">
        <span class="serif" style="display:inline-block;max-width:1500px;padding:16px 30px;border-radius:14px;
          background:rgba(21,17,14,.86);color:#f3e9d6;font-size:38px;line-height:1.3;box-shadow:0 6px 24px rgba(0,0,0,.45)">{html.escape(text)}</span></div>""",
        os.path.join(tmp, f"cap{i}.png"), transparent=True, extra_css="body{background:transparent !important}")


def overlay(frame, cap):
    if cap is None:
        return frame
    a = cap[:, :, 3:4].astype(np.float32) / 255.0
    return (frame * (1 - a) + cap[:, :, :3] * a).astype(np.uint8)


def clip_frames(name):
    """Real-time frames for a recorded scene, resampled to FPS."""
    d = os.path.join(CLIPS, name)
    meta = json.load(open(os.path.join(d, "frames.json")))
    frames = meta["frames"]
    t0, t1 = frames[0]["t"], max(meta["end"], frames[-1]["t"])
    n = int((t1 - t0) * FPS)
    cache, j = {}, 0
    for k in range(n):
        t = t0 + k / FPS
        while j + 1 < len(frames) and frames[j + 1]["t"] <= t:
            j += 1
        if j not in cache:
            cache.clear()
            img = cv2.imread(os.path.join(d, frames[j]["file"]))
            cache[j] = img if img.shape[:2] == (H, W) else cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        yield k / FPS, cache[j]


def main():
    if not glob.glob(os.path.join(CLIPS, "*", "frames.json")):
        sys.exit("no recorded scenes: run `node video/record.mjs` first")
    tmp = tempfile.mkdtemp()
    fig = lambda f: os.path.join(ROOT, "docs", f)

    # (kind, source, seconds for slides, [(start, end, caption)])
    timeline = [
        ("slide", slide(tmp, "title", "OpenCV AI Competition 2026 · powered by AWS", "ReVive",
                        "An OpenCV 5 agent that restores old photos the way a restorer would: it measures, repairs one thing at a time, checks every repair, and asks when it isn't sure.<br><br><span style='color:#e2a04f'>Samaun Mahmud</span>"), 5, []),
        ("clip", "02-hero", None, [(1, 7, "It measures six kinds of damage with OpenCV 5, then repairs one thing at a time."),
                                   (7, 13.5, "Each repair is re-measured. Restoring the faded tones made more scratches measurable, so the next repair uses the new mask."),
                                   (13.5, 23, "Colour, tones, scratches and grain repaired, and every decision explained in the log.")]),
        ("clip", "03-mask-approval", None, [(1, 9, "Here repairing would repaint 8% of the photo, so ReVive shows the damage mask and asks first."),
                                           (9.5, 16, "Approved: it fills the scratches, verifies the result, and carries on."),
                                           (16, 31, "Damage score 27 → 5.")]),
        ("clip", "04-real-slide", None, [(1, 9, "A real 1972 slide. The colours are pink, but the blacks aren't faded: ageing, or real light?"),
                                         (9, 16, "It doesn't guess. It asks."),
                                         (16, 27, "Told it's ageing, the true colours come back.")]),
        ("clip", "05-control", None, [(1, 11, "Nothing wrong? It changes nothing. Undamaged photos come back untouched.")]),
        ("clip", "06-real-portrait", None, [(1, 16, "A real 1925 portrait: tint removed, faded blacks restored.")]),
        ("clip", "07-touch-up", None, [(1, 10.5, "A real 1942 print. This scratch is too wide for the detector, so the agent leaves it."),
                                       (10.5, 21, "You can correct what it sees: paint the scratch it missed."),
                                       (21, 34, "The agent repairs your mark and checks that it now blends in."),
                                       (34, 38, "Your edit is recorded in the trace and the summary.")]),
        ("slide", slide(tmp, "arch", "Architecture", "One container on AWS Graviton",
                        "", fig("architecture.svg")), 8,
         [(0.5, 8, "CloudFront → Graviton → OpenCV 5 agent. Bedrock plans from numbers only; S3 keeps every trace.")]),
        ("slide", slide(tmp, "loop", "Agent workflow", "Perceive · plan · act · verify · ask", "", fig("agent-workflow.svg")), 8,
         [(0.5, 8, "Every action is re-measured before it's kept. Risky repairs become questions.")]),
        ("slide", slide(tmp, "quality", "Results · 35 seeded test cases", "Better than a filter chain, and harmless when nothing's wrong",
                        "", fig("figures/ssim_by_profile.png")), 7,
         [(0.5, 7, "SSIM 0.68 → 0.82 (a naive pipeline: 0.67). Undamaged photos stay at 1.00.")]),
        ("slide", slide(tmp, "verifier", "Failure handling", "The verifier catches bad decisions",
                        "", fig("figures/verifier_ablation.png")), 7,
         [(0.5, 7, "21 injected planner mistakes: 0.80 with the verifier off, 0.83 with it on. 105 simulated LLM failures: every run finished.")]),
        ("slide", slide(tmp, "truth", "Checked against ground truth", "Re-measuring finds what fading hid",
                        "", fig("figures/trace-astronaut-aged.png"), 560, centre=True), 7,
         [(0.5, 7, "After the tone step, OpenCV finds 78% of the real scratches instead of 54%.")]),
        ("slide", slide(tmp, "limits", "Honest limits", "What it doesn't do yet",
                        "<div style='text-align:left;display:inline-block'>"
                        "• Wide real scratches are missed automatically (you can paint them in)<br>"
                        "• Heavy damage is restored more conservatively than a filter chain<br>"
                        "• Strong real grain can be over-smoothed<br>"
                        "• The numbers come from synthetic damage; real scans are judged by eye</div><br><br>"
                        "<span style='color:#a39279'>All measured and written up in the report.</span>"), 7, []),
        ("slide", slide(tmp, "team", "Team", "Samaun Mahmud · solo builder",
                        "Second-year CS (AI) student.<br><br><span style='color:#e2a04f'>[replace this slide with a 10–15 s clip of yourself]</span>"), 5, []),
        ("slide", slide(tmp, "end", "Try it", "ReVive",
                        f"It repairs. It doesn't invent. It tells you exactly what it changed.<br><br>{html.escape(LIVE_URL)}<br>github.com/samaunmahmud/revive<br><br>"
                        "<span style='color:#a39279'>OpenCV 5 · Amazon Bedrock · AWS Graviton · Amazon S3 · CloudFront</span>"), 6, []),
    ]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"avc1"), FPS, (W, H))
    if not writer.isOpened():
        sys.exit("could not open an H.264 writer")
    caps, prev_tail, total = {}, None, 0

    def captions_for(i, items, t):
        for k, (a, b, text) in enumerate(items):
            if a <= t < b:
                key = (i, k)
                if key not in caps:
                    caps[key] = caption(tmp, f"{i}_{k}", text)
                return caps[key]
        return None

    for i, (kind, src, secs, items) in enumerate(timeline):
        if kind == "slide":
            frames = ((k / FPS, src) for k in range(int(secs * FPS)))
        else:
            frames = clip_frames(src)
        first = True
        for t, frame in frames:
            out = overlay(frame, captions_for(i, items, t))
            if first and prev_tail is not None:      # crossfade from the previous segment
                for f in range(int(FADE * FPS)):
                    a = (f + 1) / (FADE * FPS)
                    writer.write(cv2.addWeighted(prev_tail, 1 - a, out, a, 0))
                    total += 1
                first = False
            writer.write(out)
            total += 1
            prev_tail = out
        print(f"{kind:5s} {src if isinstance(src, str) else ''}")
    writer.release()
    print(f"wrote {OUT}: {total / FPS:.0f} s, {os.path.getsize(OUT) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
