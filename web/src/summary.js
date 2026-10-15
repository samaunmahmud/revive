// Plain-language summary of a finished trace, built only from what the trace
// records (no LLM), so it can never claim something the agent didn't do.

import { toolLabel } from './labels'

const pct = (v) => `${(v * 100).toFixed(1)}%`

function kept(step, trace, index) {
  const b = step.measured_before
  const a = step.measured_after
  switch (step.tool) {
    case 'correct_colour':
      return `Removed an ageing tint from the highlights (colour cast ${b.highlight_cast.toFixed(0)} → ${a.highlight_cast.toFixed(0)}), keeping a little natural warmth.`
    case 'enhance_contrast': {
      const revealed = Object.keys(step.revealed || {})
      const later = trace.steps.slice(index + 1).filter((s) => s.kept).map((s) => s.tool)
      let text = `Restored the faded blacks (black point ${b.black_point.toFixed(2)} → ${a.black_point.toFixed(2)}).`
      if (revealed.includes('noise_sigma') && later.includes('denoise')) text += ' That made hidden grain visible, so it was reduced next.'
      else if (revealed.length) text += ' That made more grain or marks measurable.'
      return text
    }
    case 'inpaint_scratches': {
      const e = step.mask_edits
      if (e) {
        const parts = []
        if (e.added) parts.push(`the damage you painted (${pct(e.added)} of the photo)`)
        if (e.protected) parts.push(`left the ${pct(e.protected)} you protected untouched`)
        const check = step.checks?.painted_anomaly
        return `Filled scratches and dust using your corrections: ${parts.join('; ')}.` +
          (check ? ` Your marks now blend in (stand-out ${check[0].toFixed(0)} → ${check[1].toFixed(0)}).` : '')
      }
      return `Filled scratches and dust (${pct(b.repair_area ?? b.scratch_coverage)} of the photo)${step.reason === 'approved by human' ? ', with your approval' : ''}.`
    }
    case 'denoise':
      return `Reduced film grain (noise ${b.noise_sigma.toFixed(1)} → ${a.noise_sigma.toFixed(1)}), checking it removed no more than the grain.`
    case 'sharpen':
      return 'Sharpened slightly.'
    default:
      return `${toolLabel(step.tool)}.`
  }
}

/** Returns a list of short sentences describing what the agent did and why. */
export function summarise(trace) {
  const lines = []
  trace.steps.forEach((s, i) => { if (s.kept) lines.push(kept(s, trace, i)) })

  const rejected = trace.steps.filter((s) => !s.kept)
  if (rejected.length) {
    const tools = [...new Set(rejected.map((s) => toolLabel(s.tool).toLowerCase()))]
    lines.push(`Undid ${rejected.length} attempt${rejected.length > 1 ? 's' : ''} (${tools.join(', ')}) that failed its own check, and tried gentler settings.`)
  }
  for (const h of trace.human_answers || []) {
    if (!h.approved) lines.push(`Left "${toolLabel(h.about).toLowerCase()}" alone, as you asked.`)
  }
  const open = trace.questions.filter((q) => q.about !== 'residual_damage')
  if (open.length) {
    lines.push(`Paused ${open.map((q) => toolLabel(q.about).toLowerCase()).join(' and ')} to ask you: see below.`)
  }
  if (!lines.length) lines.push('Found nothing that measured as damage, so the photo was left exactly as it was.')
  return lines
}
