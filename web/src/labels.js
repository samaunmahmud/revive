// Human-readable names for the agent's tools and OpenCV measurements.

export const TOOLS = {
  correct_colour: { label: 'Remove colour cast', short: 'Colour' },
  enhance_contrast: { label: 'Restore faded tones', short: 'Tones' },
  inpaint_scratches: { label: 'Fill scratches & dust', short: 'Scratches' },
  denoise: { label: 'Reduce grain', short: 'Grain' },
  sharpen: { label: 'Sharpen', short: 'Sharpen' },
  residual_damage: { label: 'Review the result', short: 'Review' },
}

export const toolLabel = (t) => TOOLS[t]?.label ?? t

// fmt turns a raw measurement into display text; `bad` says when it signals damage
// (mirrors the agent's thresholds in revive/agent.py).
export const METRICS = {
  scratch_coverage: {
    label: 'Scratches & dust', fmt: (v) => `${(v * 100).toFixed(2)}%`, bad: (v) => v > 0.0045,
    hint: 'share of the photo confidently marked as scratches or dust (marks that stand out from the texture around them)',
  },
  noise_sigma: {
    label: 'Grain', fmt: (v) => v.toFixed(1), bad: (v) => v > 6,
    hint: 'estimated noise level (σ)',
  },
  black_point: {
    label: 'Black point', fmt: (v) => v.toFixed(3), bad: (v) => v > 0.15,
    hint: 'darkest tones, 0–1; fading lifts blacks to grey',
  },
  highlight_cast: {
    label: 'Colour cast', fmt: (v) => v.toFixed(1), bad: (v) => v > 8,
    hint: 'tint of the brightest 10% of pixels',
  },
  contrast: {
    label: 'Tonal range', fmt: (v) => v.toFixed(3), bad: () => false,
    hint: 'usable lightness range, 0–1',
  },
  sharpness: {
    label: 'Sharpness', fmt: (v) => Math.round(v).toLocaleString(), bad: () => false,
    hint: 'variance of the Laplacian (also rises with grain)',
  },
}

export const metricLabel = (m) => METRICS[m]?.label ?? m
export const fmtMetric = (m, v) => (METRICS[m] ? METRICS[m].fmt(v) : String(v))

// Damage score for display: one decimal is enough to read the change at a glance
export const fmtScore = (v) => (v === undefined || v === null ? '' : Number(v).toFixed(1))
