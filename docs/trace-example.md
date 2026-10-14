# Annotated traces: OpenCV output changing the agent's next decision

Every number below comes from a real run (`python -m eval.trace_example`,
`python -m eval.agentic`). You can reproduce the first and third in the web app
by clicking the **Faded portrait** and **Real 1972 colour slide** samples.

## 1. Re-measurement changes what gets repaired (Faded portrait)

![trace](figures/trace-astronaut-aged.png)

| step | planner's reason (from OpenCV numbers) | verifier | what the re-measurement changed |
|---|---|---|---|
| 0 | perceive: `highlight_cast 13.6`, `black_point 0.239`, `noise_sigma 7.1`, `scratch_coverage 0.52%` (confident marks) | | starting plan: colour, then tones |
| 1 `correct_colour` 0.9 | "cast 13.58 with faded blacks (0.239) = ageing tint" | kept: cast 13.6 → 8.3, structure 0.999 | nothing new |
| 2 `enhance_contrast` clip 0 | "black_point 0.239 > 0.15: blacks faded" | kept: black point → 0.024; **noise growth 1.00×**, so the extra grain is revealed, not created | **revealed:** grain 7.1 → 10.3 and confident scratches 0.52% → 0.81% |
| 3 `inpaint_scratches` | "scratch_coverage 0.0081 > 0.0045" | kept: coverage → 0% | uses the **new** full mask |
| 4 `denoise` 10.1 | "noise_sigma 10.13 > 6.0" (strength set from the re-measured grain, not the starting 7.1) | kept: method noise 0.91× (removed about the estimated grain, not detail) | |

**Why it matters.** The damage is synthetic and seeded, so the true scratch
and dust positions are known (green in the figure). The table measures the full
repair mask; `scratch_coverage` above counts only the confident marks that
justify repairing.

| | mask size | precision (on real marks) | recall (real marks found) |
|---|---|---|---|
| mask measured at the start | 0.89% | 96% | **54%** |
| mask re-measured after the tone step | 1.78% | 79% | **78%** |

A fixed pipeline that measured once and then ran every tool would have
inpainted with the first mask and missed almost half of the real scratches. The
agent re-measures after each step, so step 3 repairs what step 2 made visible.
Step 4's strength also comes from the re-measured grain (10.1, not 7.1).

## 2. The verifier rejects a harmful step, and the retry succeeds

From the verifier ablation (`eval/agentic.py`), where a test planner injects
over-strong settings. Case: astronaut, faded.

| step | action | verifier |
|---|---|---|
| 1 | `enhance_contrast` clip 4.0 *(injected mistake)* | **rejected**: "created grain: noise grew 1.56× relative to the tones (max 1.15)". Its own target improved (black point fixed), so the old check would have kept it |
| 2 | `enhance_contrast` clip 0.6 (gentler retry) | kept: noise growth 1.05× |

Across 35 cases, the planner injected 21 mistakes. With the verifier on, mean
SSIM was 0.832 (no mistakes: 0.834). With it off, mean SSIM was 0.800. It helped
in 10 cases and made none worse.

## 3. A measurement turns into a question instead of an action (real 1972 slide)

A real public-domain colour slide from 1972 (NARA, via Wikimedia Commons). Its
dyes have faded to a strong magenta. Perception reports `highlight_cast 25.5`
with `black_point 0.067`. Aged prints usually fade their blacks as well as
tinting their highlights. Here the blacks are intact, so the tint *could* be the
real light, such as a sunset or a smoggy sky. The planner does **not** correct
it. It pauses that repair and asks the person, showing the measured evidence.

When the person answers "it's ageing", the agent re-runs with
`restore(img, answers={"correct_colour": True})`. The cast drops from 25.5 to 9.2
(structure 0.998), and natural colours return. The figure in the report's
section 6.7 shows the result.

## Honest notes

- In the first example the starting measurement already called for inpainting.
  What the re-measurement changed was **what** got inpainted (the mask) and the
  denoise strength.
- On profiles that add no scratches, the re-measurement can still rise after a
  tone step, because thin real detail gets more contrast. Earlier versions then
  inpainted that detail. Since repairs are decided from confident marks only
  (report section 6.5), no photo without marks is inpainted. The "revealed" badge
  can still appear in its trace.
