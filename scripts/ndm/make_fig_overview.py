#!/usr/bin/env python3
"""Fig. 1 of ndm.tex: study overview for npj Digital Medicine.
  (a) where the models sit in therapeutic drug monitoring: three samples of a dosing interval -> latent ODE -> AUC of this
      interval (factual) and AUC under candidate doses or regimens (counterfactual) -> dose decision
  (b) study map: the three steps, each with its question, design, data, comparators and metric, linked by the limitation
      that motivates the next step.
Schematic. Sized so that text is >= 6 pt when the figure is printed at the text width (~5 in).

  python3 make_fig_overview.py [outfile]   (default figures/overview.png)
"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os
import sys
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/figures/overview.png"
INK, GREY, ORANGE, BLUE, RED = "#111111", "#9a9a9a", "#e08a1e", "#1f5673", "#c8313a"
STEP = {1: "#555555", 2: "#12807b", 3: "#6a3d9a"}
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(7.2, 6.9))
axA = fig.add_axes([0.0, 0.645, 1.0, 0.345]); axA.set_axis_off(); axA.set_xlim(0, 100); axA.set_ylim(0, 40)
axB = fig.add_axes([0.0, 0.0, 1.0, 0.625]); axB.set_axis_off(); axB.set_xlim(0, 100); axB.set_ylim(0, 60)


def box(ax, x, y, w, h, fc, ec, lw=1.2, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.4, ms=11):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw))


def curve(t, dose=1.0, ka=1.6, ke=0.17, base=0.33):
    """steady-state-like interval: trough + absorption peak (schematic)"""
    return dose * (base + 1.9 * (np.exp(-ke * t) - np.exp(-ka * t)))


# ---------------------------------------------------------------- (a) workflow
axA.text(0.5, 38.5, "a", fontsize=11, fontweight="bold", va="top")
axA.text(4, 38.5, "Model-informed precision dosing", fontsize=9.5, va="top")   # 20/10 a co-author's titles and labels
T = np.linspace(0, 12, 200)
# sampled interval
ins1 = fig.add_axes([0.035, 0.69, 0.215, 0.215])
ins1.fill_between(T, curve(T), color=ORANGE, alpha=0.15, lw=0)
ins1.plot(T, curve(T), color=INK, lw=1.2)
rich = np.array([0, 0.33, 0.67, 1, 1.5, 2, 3, 4, 6, 9, 12])
ins1.plot(rich, curve(rich), "o", color=GREY, ms=3.2, mfc="white", mew=0.9)
for ts in (0, 1, 3):
    ins1.plot(ts, curve(ts), "o", color=ORANGE, ms=5.5, zorder=5)
ins1.set_xticks([0, 1, 3, 12]); ins1.set_yticks([]); ins1.tick_params(labelsize=7, length=2)
ins1.set_xlabel("hours after the dose", fontsize=7.5, labelpad=1)
ins1.set_title("one visit: 3 samples (0, 1, 3 h)", fontsize=8, pad=3)
for sp in ("top", "right"):
    ins1.spines[sp].set_visible(False)
ins1.text(6.5, 0.35, "AUC", color=ORANGE, fontsize=8, fontweight="bold", ha="center")
# model box
box(axA, 32, 13, 21, 15, "#f4f4f4", INK)
axA.text(42.5, 23.6, "latent ODE", fontsize=9, fontweight="bold", ha="center", va="center")
axA.text(42.5, 17.6, "data-driven\npharmacokinetic dynamics", fontsize=7.8, ha="center", va="center")
arrow(axA, 26.5, 20.5, 31.5, 20.5)
arrow(axA, 53.5, 20.5, 58.5, 20.5)
# outputs
ins2 = fig.add_axes([0.605, 0.69, 0.215, 0.215])
ins2.plot(T, curve(T), color=INK, lw=1.2, label="current dose")
ins2.fill_between(T, curve(T), color=ORANGE, alpha=0.15, lw=0)
cand_hi, = ins2.plot(T, curve(T, dose=1.35), color=BLUE, lw=1.3, ls=(0, (3, 1.5)))
cand_lo, = ins2.plot(T, curve(T, dose=0.7), color=RED, lw=1.3, ls=(0, (3, 1.5)))
ins2.set_xticks([0, 12]); ins2.set_yticks([]); ins2.tick_params(labelsize=7, length=2)
ins2.set_xlabel("hours after the dose", fontsize=7.5, labelpad=1)
ins2.set_title("predicted exposure under\ncurrent and alternative doses", fontsize=8, pad=3)
for sp in ("top", "right"):
    ins2.spines[sp].set_visible(False)
# both candidate curves in the legend 
from matplotlib.legend_handler import HandlerTuple
cur, = [l for l in ins2.get_lines() if l.get_label() == "current dose"]
ins2.legend([cur, (cand_hi, cand_lo)], ["current dose", "candidate doses"], fontsize=6.8, frameon=False, loc="upper right",
            handlelength=3.0, bbox_to_anchor=(1.27, 1.06), handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})
arrow(axA, 83.5, 20.5, 87.0, 20.5)
box(axA, 87.3, 13, 12, 15, "#fff7ec", ORANGE)
axA.text(93.3, 20.5, "inform dose\nadjustment\ntowards the\ntarget\nexposure", fontsize=7.6, ha="center", va="center")

# ---------------------------------------------------------------- (b) study map
axB.text(0.5, 59.5, "b", fontsize=11, fontweight="bold", va="top")
axB.text(4, 59.5, "Three steps addressing successive modelling limitations", fontsize=9.5, va="top")
COLS = [
    (1, "Step 1  ·  observed exposure", [
        ("Question", "AUC of the sampled interval"),
        ("Design", "latent ODE without predefined pharmacokinetic structure"),
        ("Data", "simulated cohorts with a right or wrong population model; transplant recipients"),
        ("Compared with", "Bayesian estimation (NLME), clinical estimator, Lu et al., dose-conditioned field"),
        ("Limit", "trained on single visits, the model is largely insensitive to dose changes")]),
    (2, "Step 2  ·  alternative dose", [
        ("Question", "AUC at a dose not received"),
        ("Design", "OT-FiLM: the dose moves the state, dose-independent dynamics"),
        ("Data", "two-visit simulator; cohorts with titrated doses"),
        ("Compared with", "dose-conditioned field, Lu et al."),
        ("Limit", "limited to a single dose change")]),
    (3, "Step 3  ·  alternative regimens", [
        ("Question", "AUC under other dose times and amounts"),
        ("Design", "dose-event model: doses modify drug states while preserving physiology"),
        ("Data", "regimen benchmark of the empirical Bayes model; titrated arms"),
        ("Compared with", "empirical Bayes model (EB-VAE), Lu et al."),
        ("Limit", "limited identification of nonlinear elimination from single-regimen data")]),
]
X0, W, GAP, Y0, H = 1.5, 30.0, 3.0, 1.0, 53.0
RENDERER = fig.canvas.get_renderer()
BOTTOMS = []


def height(t):
    """rendered height of a text object in data units of axB"""
    bb = t.get_window_extent(RENDERER).transformed(axB.transData.inverted())
    return bb.height

for k, (step, title, rows) in enumerate(COLS):
    x = X0 + k * (W + GAP); col = STEP[step]
    box(axB, x, Y0 + H - 6.0, W, 6.0, col, col, lw=1.6)
    axB.text(x + W / 2, Y0 + H - 3.0, title, color="white", fontsize=8.2, fontweight="bold", ha="center", va="center")
    y = Y0 + H - 8.4
    for lab, txt in rows:
        lim = lab == "Limit"
        if lim:
            axB.plot([x + 1.2, x + W - 1.2], [y + 0.9, y + 0.9], color=col, lw=0.8)
        t1 = axB.text(x + 1.2, y, lab, fontsize=7.6, fontweight="bold", color=col, va="top")
        y -= height(t1) + 0.5
        wrapped = textwrap.fill(txt, 34, break_on_hyphens=False)
        t2 = axB.text(x + 1.2, y, wrapped, fontsize=7.5, va="top", color=col if lim else INK, linespacing=1.12,
                      style="italic" if lim else "normal")
        y -= height(t2) + 1.6
    BOTTOMS.append(y)
YB = min(BOTTOMS) - 0.6                                   # boxes end just below the longest column
for k, (step, _, _) in enumerate(COLS):
    x = X0 + k * (W + GAP)
    axB.add_patch(FancyBboxPatch((x, YB), W, Y0 + H - YB, boxstyle="round,pad=0,rounding_size=1.2", fc="none",
                                 ec=STEP[step], lw=1.6, zorder=0))
for k in range(2):
    x = X0 + (k + 1) * W + k * GAP
    arrow(axB, x - 0.3, Y0 + H - 3.0, x + GAP + 0.3, Y0 + H - 3.0, color=INK, lw=1.6, ms=11)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
from PIL import Image, ImageOps                              # trim the white margin left by the invisible axes
im = Image.open(OUT).convert("RGB"); bb = ImageOps.invert(im).getbbox()
pad = 12; im.crop((max(bb[0] - pad, 0), max(bb[1] - pad, 0), min(bb[2] + pad, im.size[0]), min(bb[3] + pad, im.size[1]))).save(OUT, dpi=(300, 300))
print("wrote", OUT)
