"""Charts for the technical report, from eval/results (run evaluate.py and agentic.py first).

    python -m eval.charts            # writes docs/figures/*.png

1. ssim_by_profile.png   - restoration quality per damage type: damaged / fixed / ReVive / ReVive+human
2. verifier_ablation.png - per-case SSIM with injected planner mistakes, verifier off vs on
3. outcomes.png          - how the 35 runs end: auto-complete vs which question was asked
4. latency.png           - per-tool latency (median, p95)
Colours: validated categorical palette (fixed order); the raw damaged input is neutral grey.
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = os.path.join("eval", "results")
OUT = os.path.join("docs", "figures")

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
GREY = "#b9b8b2"
FIXED, REVIVE, HUMAN = "#eb6834", "#2a78d6", "#1baf7a"   # categorical slots 2, 1, 3
ASK_MASK, ASK_COLOUR = "#eda100", "#6250d6"               # slots 4, 7: only used for question types

PROFILES = ["none", "faded", "sepia", "grainy", "scratched", "aged", "heavy"]
TOOL_NAMES = {"correct_colour": "Remove colour cast", "enhance_contrast": "Restore faded tones",
              "inpaint_scratches": "Fill scratches & dust", "denoise": "Reduce grain", "sharpen": "Sharpen"}


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 10, "text.color": INK,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.titlelocation": "left", "axes.titlepad": 12,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 9,
    })


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def ssim_by_profile():
    with open(os.path.join(RESULTS, "results.csv")) as fh:
        rows = list(csv.DictReader(fh))
    series = [("Damaged input", "ssim_damaged", GREY), ("Naive fixed pipeline", "ssim_fixed", FIXED),
              ("ReVive (automatic)", "ssim_revive", REVIVE), ("ReVive + human answers", "ssim_revive_hitl", HUMAN)]
    means = {k: [np.mean([float(r[k]) for r in rows if r["damage"] == p]) for p in PROFILES] for _, k, _ in series}

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x, w = np.arange(len(PROFILES)), 0.19
    for i, (label, key, color) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, means[key], w - 0.03, color=color, label=label)
    # label only the headline: ReVive on the undamaged control ("do no harm")
    ax.annotate(f"{means['ssim_revive'][0]:.2f}", (x[0] + 0.5 * w, means["ssim_revive"][0]),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.annotate(f"{means['ssim_fixed'][0]:.2f}", (x[0] - 0.5 * w, means["ssim_fixed"][0]),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, ["undamaged\n(control)"] + PROFILES[1:])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("SSIM vs clean original (higher is better)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Restoration quality by damage type (5 images each)")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    save(fig, "ssim_by_profile.png")


def verifier_ablation(data):
    on, off = data["verifier_on"], data["verifier_off"]
    s = data["summary"]["verifier_ablation"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for a, b in zip(off, on):
        changed = abs(a["ssim"] - b["ssim"]) > 0.005
        ax.plot([0, 1], [a["ssim"], b["ssim"]], color=REVIVE if changed else GREY,
                lw=2 if changed else 1, alpha=1 if changed else 0.6, marker="o", ms=4 if changed else 2.5,
                zorder=3 if changed else 2)
    ax.set_xticks([0, 1], [f"verifier OFF\nmean {s['verifier_off']['mean_ssim']:.3f}",
                           f"verifier ON\nmean {s['verifier_on']['mean_ssim']:.3f}"])
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("SSIM vs clean original")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"Verifier vs {s['mistakes_injected']} injected planner mistakes")
    ax.text(1.02, 0.02, f"blue: {s['cases_where_verifier_helped']} cases improved by\nverification; "
                        f"{s['cases_where_verifier_hurt']} made worse", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=INK_2)
    save(fig, "verifier_ablation.png")


def outcomes(data):
    b = data["summary"]["baseline"]
    n = data["summary"]["cases"]
    auto = round(b["auto_complete_rate"] * n)
    reasons = b["escalations_by_reason"]
    parts = [("completed automatically", auto, REVIVE),
             ("asked: approve scratch mask", reasons.get("inpaint_scratches", 0), ASK_MASK),
             ("asked: ageing or real colour?", reasons.get("correct_colour", 0), ASK_COLOUR)]
    fig, ax = plt.subplots(figsize=(8, 1.9))
    left = 0
    for label, v, color in parts:
        ax.barh(0, v, left=left, color=color, height=0.5, edgecolor=SURFACE, linewidth=2)
        ax.text(left + v / 2, 0, str(v), ha="center", va="center", color=INK if color == ASK_MASK else "white",
                fontsize=10, fontweight="bold")
        left += v
    ax.set_xlim(0, n)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"runs (of {n})")
    ax.set_title("How the agent's runs end")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts], [p[0] for p in parts],
              ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.55))
    save(fig, "outcomes.png")


def latency(data):
    lat = data["summary"]["latency_ms"]
    tools = sorted(lat, key=lambda t: lat[t]["median"])
    med = [lat[t]["median"] for t in tools]
    p95 = [lat[t]["p95"] for t in tools]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    y = np.arange(len(tools))
    ax.barh(y, med, height=0.45, color=REVIVE)
    ax.hlines(y, med, p95, color=INK_2, lw=1)
    ax.plot(p95, y, "|", color=INK_2, ms=10)
    for yi, m in zip(y, med):
        ax.text(m, yi + 0.33, f"{m:.0f} ms", fontsize=8, color=INK)
    ax.set_yticks(y, [TOOL_NAMES.get(t, t) for t in tools])
    ax.set_xlabel("ms per step on this machine (bar: median, line: to p95; includes re-measurement)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Per-step latency (~512 px photos)")
    save(fig, "latency.png")


def main():
    style()
    with open(os.path.join(RESULTS, "agentic.json")) as fh:
        data = json.load(fh)
    ssim_by_profile()
    verifier_ablation(data)
    outcomes(data)
    latency(data)


if __name__ == "__main__":
    main()
