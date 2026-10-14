# ReVive — agentic photo restoration with OpenCV 5

ReVive restores old, damaged photos the way a human restorer would: it **measures**
what is wrong, **plans** a repair, **acts**, then **re-measures** to check the repair
actually helped. When a fix could change what the photo really shows, it **asks a
human** instead of guessing.

Entry for the OpenCV AI Competition 2026 (powered by AWS): Agentic Vision with OpenCV 5.

**Live demo:** _pending AWS deployment_ · **Report:** [docs/REPORT.md](docs/REPORT.md) ·
**Annotated traces:** [docs/trace-example.md](docs/trace-example.md)

![agent workflow](docs/agent-workflow.svg)

## Why it's an agent, not a filter chain

OpenCV 5 measurements decide what happens next, and they are taken again after
every step:

- **Re-measurement changes the repair.** On the "Faded portrait" sample, restoring
  the faded tones lets OpenCV find **78%** of the true scratches instead of **54%**,
  so inpainting uses the new mask. The denoise strength also comes from the
  re-measured grain.
- **The verifier rejects harmful steps.** A step is kept only if:
  - its own metric improved,
  - structure survived (blurred SSIM),
  - contrast didn't create grain,
  - denoising didn't remove detail.

  Otherwise it is retried 40% gentler (at most twice), and then the agent gives
  up on that tool.
- **It asks instead of guessing.** A colour cast with intact blacks may be the
  real scene (coffee is brown), so it asks. Damage over 3% of the photo means a
  person approves the red damage mask before anything is filled in.
- **Every step is traced:** the measurements before and after, the reason, the
  parameters, the checks, the verdict and the time. The trace streams live to the
  UI and is archived in S3.

## Results (35 seeded cases: 5 public-domain images × 7 damage types)

| | SSIM | PSNR (dB) |
|---|---|---|
| Damaged input | 0.680 | 25.1 |
| Naive fixed pipeline (every tool once) | 0.672 | 18.8 |
| **ReVive (automatic)** | **0.822** | **29.4** |
| ReVive + simulated human answers | 0.834 | 29.5 |

- **Do no harm:** on undamaged photos ReVive applies **no steps at all** (SSIM
  1.000); the naive pipeline drops them to 0.75. It also never inpaints a photo
  that has no scratches (0 of 10 inpaint steps, down from 9 of 19 before the
  confident-marks check).
- **Verifier vs planner mistakes:** with 21 injected over-strong decisions, mean
  SSIM is 0.800 with the verifier off and **0.832** with it on (0.834 with no
  mistakes). It helped in 10 cases and hurt in none.
- **LLM failure handling:** across 105 simulated Bedrock failures (timeouts,
  broken JSON, unknown and blocked tools), **35/35** runs completed, with results
  identical to the heuristic planner's.
- **Outcomes:** 26/35 runs complete automatically; 9 ask a person (5 mask
  approvals, 4 colour questions).
- **Known weak spots:**
  - heavy damage, where the naive pipeline scores 0.51 vs 0.43 automatic and 0.51
    with approval;
  - the scratch threshold is tuned on synthetic, high-contrast marks, so faint
    real scratches may be missed;
  - synthetic damage only so far.

  See [docs/REPORT.md §7](docs/REPORT.md#7-limitations).

![quality by damage type](docs/figures/ssim_by_profile.png)

## Quick start

```bash
pip install -r requirements.txt                       # Python 3.12+; OpenCV 5 pinned
python restore.py old_photo.jpg                       # CLI, offline heuristic planner
python restore.py old_photo.jpg --planner bedrock     # LLM planner on Amazon Bedrock
python restore.py old_photo.jpg --approve inpaint_scratches --decline correct_colour
```

### Web app

```bash
cd web && npm install && npm run build && cd ..
uvicorn server.app:app --port 8000                    # open http://localhost:8000
```

The UI streams the agent's trace live (Server-Sent Events). It shows the damage
mask when the agent asks for approval, and it re-runs the agent with your answers
through `restore(img, answers=...)`. Five demo photos ship in `server/samples`
(regenerate them with `python -m server.make_samples`). For frontend development,
run `uvicorn server.app:app --reload` alongside `npm run dev` in `web/`; Vite on
:5173 proxies `/api` to :8000.

The Bedrock planner needs AWS credentials, plus optional `AWS_REGION` and
`REVIVE_MODEL_ID`. If a model call fails, the agent falls back to the heuristic
planner and says so in the trace.

## Deploy on AWS (Graviton)

![architecture](docs/architecture.svg)

Everything is one CloudFormation stack (`deploy/cloudformation.yaml`):
- a VPC and a Graviton instance (IMDSv2 only, no SSH; admin through SSM Session
  Manager);
- an Elastic IP;
- CloudFront (caching off for the API and the live stream, on for hashed assets);
- a private, encrypted S3 bucket with 30-day expiry;
- an instance role scoped to that bucket, Bedrock and ECR pull.

Prerequisites:
- AWS CLI v2 with credentials (`aws configure`);
- Docker running;
- Bedrock model access for Anthropic models in your region.

```bash
python deploy/check_bedrock.py --only haiku     # which Claude IDs work on this account
REVIVE_MODEL_ID=<working id> ./deploy/deploy.sh # build ARM64 image → ECR → stack; prints the URL
./deploy/deploy.sh                              # later: push a new image and restart (~2 min)
./deploy/teardown.sh                            # delete everything (stack, bucket contents, ECR)
```

Defaults are region `eu-west-2` (`AWS_REGION`) and instance `t4g.small`
(`REVIVE_INSTANCE_TYPE`). While it runs, it costs about $0.60 a day:
- t4g.small ≈ $0.019/h;
- public IPv4 ≈ $0.005/h;
- CloudFront and S3 stay within free-tier volumes for a demo.

Run `teardown.sh` when you don't need it.

The same image runs locally:
`docker build --platform linux/arm64 -t revive . && docker run -p 8000:8000 revive`.

## Reproducing the evaluation

```bash
python -m eval.evaluate        # quality vs damaged input and naive pipeline → eval/results/
python -m eval.agentic         # success/escalation, verifier ablation, LLM-failure test
python -m eval.trace_example   # annotated trace with ground-truth scratch positions
python -m eval.charts          # figures → docs/figures/
python -m eval.planners        # Bedrock vs heuristic planner (needs AWS credentials)
python -m bench.benchmark run --label stock --price <$/h>   # OpenCV latency/throughput/cost
python -m bench.benchmark compare bench/results/*.json      # stock vs COOL table
```

## Project layout

```
revive/perception.py   OpenCV 5 measurements (the agent's eyes)
revive/tools.py        OpenCV 5 repair tools (the agent's hands)
revive/agent.py        plan → act → verify loop, planners, side-effect checks, human questions, trace
revive/damage.py       seeded synthetic ageing for evaluation
restore.py             command-line entry point
server/                FastAPI: jobs, live SSE trace, human answers, S3 archive, demo samples
web/                   React + Vite frontend (darkroom UI)
eval/                  evaluate (quality), agentic (behaviour), planners, trace_example, charts
bench/                 reproducible OpenCV benchmark (stock vs COOL)
deploy/                CloudFormation stack, deploy/teardown scripts, Bedrock model check
docs/                  report, diagrams, annotated traces, figures, video script
Dockerfile             ARM64 image: builds the UI, runs FastAPI + the agent
```

## Status (deadline 26 Oct, 11:59pm PT)

- [x] OpenCV 5 perception and repair tools, agent loop, verifier with side-effect checks, trace
- [x] Evaluation: quality, agentic metrics, verifier ablation, LLM failure handling, charts
- [x] Web app: upload, live trace, approve/decline with mask evidence, before/after slider
- [x] ARM64 container and one-stack AWS deployment (CloudFront, Graviton, S3, Bedrock)
- [x] Report draft, architecture and agent-workflow diagrams, annotated traces, video script
- [ ] Deploy live and run the Bedrock-vs-heuristic evaluation
- [ ] COOL vs stock OpenCV benchmark on Graviton4
- [ ] Real scanned old photos in the evaluation
- [ ] Record the video

## Responsible use

ReVive repairs damage; it does not invent content. It uses classical OpenCV
operations only, with no generative fill. Large-area inpainting and ambiguous
colour changes need human approval, and every change is logged with its evidence.
The LLM planner receives six numbers, never the image. Restored images should be
labelled as restored when used as historical records.

Sample images are public-domain photos from scikit-image.
