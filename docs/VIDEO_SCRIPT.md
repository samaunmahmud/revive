# ReVive demo video script (target 4:45, hard limit 5:00)

Best material first: judges may stop early. The live agent loop is the hook.
Record the app at 1440×900 in Chrome (deployed URL, so the address bar shows
CloudFront), voice-over recorded separately. **[bracketed]** = what's on screen.

---

### 0:00–0:25 · Hook: the agent at work
**[Click "Faded portrait". The loop strip lights up Perceive → Plan → Act → Verify; log cards appear one by one.]**

> "This is ReVive. It restores old photos the way a restorer would. It measures the
> damage with OpenCV 5, makes one repair, then measures again to check the repair
> actually helped. Watch step two."

**[Zoom on step 2 card: "revealed hidden grain / scratches" badges.]**

> "Restoring the faded tones made scratches visible that weren't measurable
> before. So the next repair uses the new measurement. That's the difference
> between an agent and a filter chain."

### 0:25–0:50 · Result
**[Drag the before/after slider slowly. Point at the damage score 4.6 → 0.0.]**

> "Colour cast removed, tones restored, scratches filled, grain reduced, with every
> decision explained in the log."

### 0:50–2:05 · Asks instead of guessing, and lets you correct it (human control)
**[Click "Badly damaged cat". The agent pauses; the question card shows the red damage mask.]**

> "Here almost ten percent of the photo is damaged. Filling that much means
> inventing pixels, so ReVive shows me exactly what it wants to repair and asks
> first."

**[Click "Approve repair mask". The re-run streams; score 27.0 → 5.0.]**

**[Click "Real 1972 colour slide" (a real public-domain scan). The colour question appears; click "It's ageing, remove it"; drag the slider.]**

> "This one is a real slide from 1972, and it's subtler. The colours are pink,
> but the blacks aren't faded, and aged prints usually fade both. So the tint
> could be real light. ReVive asks. I tell it it's ageing, and the true colours
> come back."

**[Click "Real 1942 scratched print". The agent finds nothing. Click "Touch up damage", paint one stroke along the scratch, click "Repair with my corrections"; drag the slider; point at the "your mask" badge in the log.]**

> "And when it misses something, you can correct what it sees. This 1942 scratch
> is too wide for its detector, so it left it alone. I paint over it. The agent
> repairs my mark, then measures that it now blends in with its surroundings,
> and the trace records that the edit was mine. On test photos, one painted
> stroke cuts the error inside a missed scratch from 119 to 15."

### 2:05–2:25 · Do no harm
**[Click "Undamaged control". Zero steps; "Restored" with 0 repairs.]**

> "And when nothing is wrong, it does nothing. On undamaged photos ReVive makes
> zero changes. A naive enhance-everything pipeline drops them to 75 percent
> structural similarity. And it never repaints a photo that has no scratches."

### 2:25–3:05 · Architecture
**[Show docs/architecture.svg, then docs/agent-workflow.svg; highlight boxes as you speak.]**

> "Everything runs on AWS. A single container sits on a Graviton ARM instance
> behind CloudFront, and finished jobs are archived in S3. The planner can be an
> LLM on Amazon Bedrock, but it never sees the photo, only OpenCV's six
> measurements. If Bedrock fails, a deterministic planner takes over, and the
> trace says so. OpenCV 5 does all the vision: perception, the five repair tools
> and the verifier."

**[If COOL is done:]** "The image processing runs on Cloud-Optimized OpenCV for
Graviton. Here's what it bought us." **[benchmark table]**

### 3:05–4:05 · Results
**[docs/figures/ssim_by_profile.png]**

> "On 35 test cases, ReVive lifts structural similarity from 0.68 to 0.82. A fixed
> pipeline actually makes things worse, at 0.67."

**[docs/figures/verifier_ablation.png]**

> "To test the verifier, we gave the agent a planner that makes bad choices,
> like over-strong contrast and over-strong denoising. With the verifier off,
> quality drops to 0.80. With it on, the agent rejects those steps, retries
> gently, and gets 0.83, almost the same as with no mistakes. We also hit the LLM
> planner with 105 simulated failures, and every restoration still finished."

**[docs/figures/trace-astronaut-aged.png]**

> "And because our test damage is synthetic, we can check the agent against the
> truth. Re-measuring after the tone step found 78 percent of the real scratches,
> up from 54."

### 4:05–4:30 · Honest limits
**[Bulleted slide]**

> "It isn't perfect. Heavy damage is restored more conservatively than an
> aggressive filter chain, and faint scratches in busy texture can be missed. And
> most of our numbers come from synthetic damage. It's all in the report."

### 4:30–4:45 · Team and close
**[Face cam or photo + name card: Samaun Mahmud, solo, second-year CS (AI)]**

> "I'm Samaun, and I built ReVive solo. It repairs, it doesn't invent, and it
> tells you exactly what it changed. Try it at the link below."

**[End card: live URL · GitHub repo · "OpenCV 5 · Amazon Bedrock · AWS Graviton"]**

---

## Recording checklist
- [ ] Deployed URL working; Bedrock toggle state decided (show "Bedrock LLM" if live)
- [ ] Browser zoom 110%, no bookmarks bar, clean profile
- [ ] Each sample clicked once beforehand (warm cache; the live pacing still shows the steps)
- [ ] Voice-over: quiet room, read slowly. 4:30 of script runs ≈ 600 words
- [ ] Captions burned in (judges often watch muted)
- [ ] Upload to YouTube as public or unlisted; check it plays logged-out
