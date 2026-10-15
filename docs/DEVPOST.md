# Devpost submission text (draft)

Copy each section into the matching Devpost field. Items marked **[AWS]** wait
for the deployment and the Bedrock and COOL runs; replace them before submitting.
Every number here comes from `docs/REPORT.md` and the scripts in `eval/`.

---

## Project name

ReVive

## Tagline

An OpenCV 5 agent that restores old photos like a careful restorer: it measures,
repairs one thing at a time, checks every repair, and asks when it isn't sure.

## Inspiration

Family albums, local-history societies and small museums hold millions of prints
that are fading, yellowing and scratched. Restoring them by hand is slow, skilled
work. One-click "AI enhance" tools go wrong in the other direction: they run every
filter on every photo, and generative models can invent faces and details. For a
family record or an archive, an invented detail is worse than a scratch.

A good restorer works differently. They look first, fix one problem at a time,
check that each fix helped, and stop to ask when a change could alter what the
photo really shows. We wanted an agent that works that way, with every decision
grounded in measurements and written down.

## What it does

You upload a scan or click one of six sample prints, three of them real
public-domain scans. You then watch the agent work live:

- **Perceive.** OpenCV 5 measures six kinds of damage: scratch and dust coverage,
  grain, faded blacks, tonal range, colour cast and sharpness.
- **Plan.** A planner chooses one next repair and its settings, or decides to
  ask. It is either a deterministic heuristic or Claude on Amazon Bedrock, which
  sees only the numbers, never the pixels.
- **Act.** It applies one OpenCV 5 tool: colour correction, tone restoration,
  inpainting, denoising or sharpening.
- **Verify.** It re-measures the photo. A step is kept only if its own metric
  improved, the structure survived, contrast didn't create grain, and denoising
  didn't remove detail. Otherwise it retries 40% gentler, then gives up on that
  tool.
- **Ask, don't guess:**
  - A colour cast with intact blacks might be the real scene, so it asks.
  - Repainting more than 3% of the photo needs you to approve the red damage
    mask first.
- **Let you correct it.** When the agent misses a scratch or flags real detail,
  you paint over the miss or protect the detail with a brush. The agent applies
  your corrections at repair time and checks that every painted mark now blends
  in.
- **Trace.** Every step is logged with its measurements before and after, the
  reason, the verdict and the timing. You can download it next to the restored
  photo, and it is archived in S3.

**Results on 35 seeded test cases** (5 public-domain images × 7 damage types,
scored against the clean original):
- Mean SSIM goes from **0.680 to 0.822**. A naive "run every filter" pipeline
  gets 0.672.
- Undamaged photos are left untouched (**SSIM 1.000**). The naive pipeline drops
  them to 0.75.
- When we injected 21 bad planner decisions, the verifier recovered almost all
  the loss: 0.800 with the verifier off, **0.832** with it on.
- In 105 simulated LLM failures, every restoration still finished.
- For a scratch the detector cannot see, one painted stroke cuts the error inside
  it from **118.6 to 14.7**.

## How we built it

- **OpenCV 5** (`opencv-python-headless==5.0.0.93`) does all the perception and
  every repair:
  - median-residual scratch detection with shape rules and a per-mark contrast
    test;
  - Immerkaer noise estimation;
  - LAB colour statistics;
  - Telea and Navier-Stokes inpainting, non-local-means denoising, CLAHE.
- **The agent loop** is under 500 lines of Python. It has no framework, so each
  decision is easy to trace and test.
- **Amazon Bedrock** (Claude via the Converse API) is the optional planner. Its
  replies are checked and clamped. If a call fails, that step falls back to the
  heuristic and the trace says why.
- **The web app:** a FastAPI backend streams the trace over Server-Sent Events
  to a React + Vite frontend, with a darkroom look. The mask editor is a canvas
  brush that sends PNG layers to the agent.
- **AWS:** one CloudFormation stack runs the ARM64 container on EC2 Graviton
  behind CloudFront, archives every job to S3, and pulls images from ECR.
  **[AWS: instance type, live URL]**
- **COOL:** **[AWS: the Cloud-Optimized OpenCV benchmark on Graviton4, or remove
  this line]**
- **Testing:** 39 pytest tests run in GitHub Actions, and there are reproducible
  evaluation scripts for every number above.

## Challenges we ran into

- **Telling scratches from real detail.** Tripod legs, tower lattices and
  hair look like scratches to a simple detector. We added a per-mark contrast
  test that decides whether to repair, while the full mask decides what to
  repaint. Inpainting on photos with no marks fell from 9 of 19 steps to 0 of 10.
- **A verifier that only checks its own metric is not enough.** Over-strong
  contrast "fixed" the faded blacks but manufactured grain. We added side-effect
  checks, and the ablation shows they matter.
- **Fixing one thing reveals another.** Restoring faded tones makes hidden grain
  and scratches measurable, so measuring once is wrong. Re-measuring after the
  tone step finds 78% of the true scratches instead of 54%.
- **Real scans are harder than synthetic damage.** A real 1942 print has a wide
  scratch the detector can't see. That led us to build mask corrections instead
  of tuning the detector until it found false positives everywhere.
- **Honest evaluation.** We publish failure cases: heavy damage is restored more
  conservatively than a filter chain, and strong real grain can be over-smoothed.

## Accomplishments that we're proud of

- **"Do no harm" is measurable:** undamaged photos come back bit-identical.
- **OpenCV output visibly changes the agent's next decision,** with an annotated
  trace checked against ground truth.
- **A person can correct the agent's perception, not just approve it,** and the
  agent verifies the correction.
- **Every claim in the report has a script that reproduces it.**

## What we learned

The most valuable part of the agent turned out not to be the planner but the
verifier and the questions. A model that plans from six numbers is only safe
because every action is re-measured and every risky one is handed to a person.

## What's next for ReVive

- A detector for wide scratches and emulsion flecks, trained and validated on
  real scans.
- Texture-aware denoising, so strong real grain isn't smoothed to wax.
- Batch mode for archives: restore a folder, and queue only the questions for a
  person.
- Restoration metadata embedded in the output file, so a restored photo always
  carries its trace.

## Built with

opencv, python, fastapi, react, vite, amazon-bedrock, amazon-ec2, aws-graviton,
amazon-s3, amazon-cloudfront, amazon-ecr, aws-cloudformation, docker, claude

## Links

- Live demo: **[AWS: CloudFront URL]**
- Code: https://github.com/samaunmahmud/revive (make public before submitting)
- Video: **[YouTube/Vimeo link, max 5 minutes]**
- Technical report: `docs/ReVive-report.pdf` in the repo
