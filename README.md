# ReVive — agentic photo restoration with OpenCV 5

ReVive restores old, damaged photos the way a human restorer would: it **measures**
what is wrong, **plans** a repair, **acts**, then **re-measures** to check the repair
actually helped. When a fix could change what the photo really shows, it **asks a
human** instead of guessing.

Entry for the OpenCV AI Competition 2026 (powered by AWS) — Agentic Vision path.

## How it works

```
            ┌─────────────── perceive (OpenCV 5) ───────────────┐
photo ──▶   │ scratch_coverage · noise_sigma · black_point ·     │
            │ highlight_cast · contrast · sharpness              │
            └──────────────────────┬─────────────────────────────┘
                                   ▼
                     plan (Bedrock LLM / heuristic)
                     picks ONE tool + settings, or asks a human
                                   ▼
            act (OpenCV 5): correct_colour · enhance_contrast ·
                            inpaint_scratches · denoise · sharpen
                                   ▼
            verify (OpenCV 5): did the target metric improve?
                               is the structure (SSIM) preserved?
                 kept ─────────────┤──────────── rejected → retry 40% gentler
                                   │                       (max 2), then give up
                                   ▼
                         loop until done / human review
```

**Why it's agentic, not a fixed pipeline** — the OpenCV measurements change what
happens next:

- Restoring faded tones often *reveals* scratches and grain that were invisible
  before. The agent logs them as revealed and repairs them in later steps.
- A repair that smears detail (structure below threshold) is rejected and retried
  with gentler settings or a different algorithm.
- A colour cast with intact blacks may be the real scene (a cup of coffee is brown),
  so the agent asks rather than "correcting" it. Large damaged areas (>3% of the
  photo) need a human to approve the repair mask before inpainting.
- Every step is recorded in a trace (measurements before/after, decision, verdict).

## Quick start

```bash
pip install -r requirements.txt
python restore.py old_photo.jpg                       # offline heuristic planner
python restore.py old_photo.jpg --planner bedrock     # LLM planner on Amazon Bedrock
python restore.py old_photo.jpg --approve inpaint_scratches --decline correct_colour
```

Bedrock needs AWS credentials, plus optional `AWS_REGION` and `REVIVE_MODEL_ID`. If
the model call fails, the agent falls back to the heuristic planner and notes it
in the trace.

## Evaluation

`python -m eval.evaluate` takes 5 public-domain images, applies 7 seeded damage
profiles (including an undamaged control), and compares three things against the
clean original:

- the damaged input,
- a naive fixed pipeline (every tool once, default settings),
- ReVive.

| (35 cases) | SSIM | PSNR (dB) |
|---|---|---|
| Damaged input | 0.680 | 25.1 |
| Fixed pipeline | 0.672 | 18.8 |
| **ReVive (no human)** | **0.818** | **27.3** |
| ReVive + simulated human answers | 0.824 | 27.4 |

- 26/35 cases complete automatically (SSIM 0.885); 9 are escalated to a human.
- On undamaged control photos ReVive leaves the image essentially untouched
  (SSIM ≥ 0.98), while the fixed pipeline drops them to about 0.74.

Outputs go to `eval/results/`: `results.csv`, `traces.json`, and side-by-side
images (damaged | ReVive | original).

### Known limitations (found by the evaluation)

- **Warm subjects with a tint.** A naturally warm subject (orange cat fur) under an
  ageing tint gets slightly over-corrected (chelsea/sepia: 0.885 vs 0.939 untouched).
- **Scratches over similar tones.** Where a scratch crosses a similar-toned
  background, it breaks into short fragments that fail the "long and thin" rule, so
  faint pieces remain. Planned fix: join collinear fragments (Hough line grouping)
  before the length test.
- **Thin real detail.** Very thin real features (tripod legs, spoon edges) can be
  flagged as scratches; inpainting them is mostly invisible but not lossless.
- **Synthetic damage only.** The evaluation uses synthetic ageing; real scanned
  prints are still to be added.

## Project layout

```
revive/perception.py   OpenCV 5 measurements (the agent's eyes)
revive/tools.py        OpenCV 5 repair tools (the agent's hands)
revive/agent.py        plan → act → verify loop, planners, human questions, trace
revive/damage.py       seeded synthetic ageing for evaluation
eval/evaluate.py       benchmark vs damaged input and fixed pipeline
restore.py             command-line entry point
```

## Roadmap to submission (deadline 26 Oct, 11:59pm PT)

- [x] OpenCV 5 perception and repair tools, agent loop, verifier, trace
- [x] Evaluation harness with baselines and failure cases
- [ ] Web app: upload, live step-by-step trace, approve/decline questions, before/after slider
- [ ] AWS: API on Graviton (ECS or EC2), S3 for images and traces, Bedrock planner live
- [ ] COOL benchmark on Graviton vs stock OpenCV (latency per tool) for the COOL award
- [ ] Real scanned old photos in the evaluation
- [ ] Technical report, architecture and agent-workflow diagrams, video under 5 minutes

## Responsible use

ReVive repairs damage; it does not invent content. Large-area inpainting and
ambiguous colour changes need human approval, and every change is logged. Restored
images should be labelled as restored when used as historical records.
