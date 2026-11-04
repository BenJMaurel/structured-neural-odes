#!/usr/bin/env python3
"""Fig. 1 of the npj rewrite: study overview.
  (a) where the models sit in therapeutic drug monitoring: unchanged from ../make_fig_overview.py (a co-author's titles and labels)
  (b) the five questions of the study, in the order of the Results, with the model, the test data and the comparators of each.
Schematic. The figure is printed at 0.95 of the text width, so that the text stays at least 6 pt.

  python3 make_fig_overview.py outfile
"""
import os
import sys
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.legend_handler import HandlerTuple

OUT = sys.argv[1]
INK, GREY, ORANGE, BLUE, RED = "#111111", "#9a9a9a", "#e08a1e", "#1f5673", "#c8313a"
MODEL = {"latent": "#555555", "shift": "#12807b", "event": "#6a3d9a", "both": "#1f5673"}
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})

FW, FH = 7.2, 4.55                      # inches
TOP_IN = 2.38                           # height of panel a, unchanged in inches
A_BOT = FH - TOP_IN - 0.05
fig = plt.figure(figsize=(FW, FH))
axA = fig.add_axes([0.0, A_BOT / FH, 1.0, TOP_IN / FH]); axA.set_axis_off(); axA.set_xlim(0, 100); axA.set_ylim(0, 40)
axB = fig.add_axes([0.0, 0.0, 1.0, (A_BOT - 0.08) / FH]); axB.set_axis_off(); axB.set_xlim(0, 100); axB.set_ylim(0, 100)
INS_Y, INS_H = (A_BOT + 0.31) / FH, 1.4835 / FH


def box(ax, x, y, w, h, fc, ec, lw=1.2, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.4, ms=11):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw))


def curve(t, dose=1.0, ka=1.6, ke=0.17, base=0.33):
    """steady-state-like interval: trough + absorption peak (schematic)"""
    return dose * (base + 1.9 * (np.exp(-ke * t) - np.exp(-ka * t)))


# ---------------------------------------------------------------- (a) workflow
axA.text(0.5, 38.5, "a", fontsize=11, fontweight="bold", va="top")
axA.text(4, 38.5, "Model-informed precision dosing", fontsize=9.5, va="top")
T = np.linspace(0, 12, 200)
ins1 = fig.add_axes([0.035, INS_Y, 0.215, INS_H])
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
box(axA, 32, 13, 21, 15, "#f4f4f4", INK)
axA.text(42.5, 23.6, "latent ODE", fontsize=9, fontweight="bold", ha="center", va="center")
axA.text(42.5, 17.6, "data-driven\npharmacokinetic dynamics", fontsize=7.8, ha="center", va="center")
arrow(axA, 26.5, 20.5, 31.5, 20.5)
arrow(axA, 53.5, 20.5, 58.5, 20.5)
ins2 = fig.add_axes([0.605, INS_Y, 0.215, INS_H])
cur, = ins2.plot(T, curve(T), color=INK, lw=1.2)
ins2.fill_between(T, curve(T), color=ORANGE, alpha=0.15, lw=0)
cand_hi, = ins2.plot(T, curve(T, dose=1.35), color=BLUE, lw=1.3, ls=(0, (3, 1.5)))
cand_lo, = ins2.plot(T, curve(T, dose=0.7), color=RED, lw=1.3, ls=(0, (3, 1.5)))
ins2.set_xticks([0, 12]); ins2.set_yticks([]); ins2.tick_params(labelsize=7, length=2)
ins2.set_xlabel("hours after the dose", fontsize=7.5, labelpad=1)
ins2.set_title("predicted exposure under\ncurrent and alternative doses", fontsize=8, pad=3)
for sp in ("top", "right"):
    ins2.spines[sp].set_visible(False)
ins2.legend([cur, (cand_hi, cand_lo)], ["current dose", "candidate doses"], fontsize=6.8, frameon=False, loc="upper right",
            handlelength=3.0, bbox_to_anchor=(1.27, 1.06), handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})
arrow(axA, 83.5, 20.5, 87.0, 20.5)
box(axA, 87.3, 13, 12, 15, "#fff7ec", ORANGE)
axA.text(93.3, 20.5, "inform dose\nadjustment\ntowards the\ntarget\nexposure", fontsize=7.6, ha="center", va="center")

# ---------------------------------------------------------------- (b) the five questions
axB.text(0.5, 99.5, "b", fontsize=11, fontweight="bold", va="top")
axB.text(4, 99.5, "Five questions, in the order of the Results", fontsize=9.5, va="top")
COLX = [5.0, 58.0]
HEAD = ["Question", "Evaluated on"]
ROWS = [
    ("latent", "Is accuracy at the current dose enough?", "simulation and real patients"),
    ("shift", "What does a model need to predict an alternative dose?", "dose-shift model, tacrolimus simulation"),
    ("event", "Can it handle missed or loading doses and new schedules?", "dose-event model, regimen simulations"),
    ("event", "Does it learn how a dose works, or who gets which dose?", "doses adjusted to the patient, simulation"),
    ("both", "Does it carry over to real patients?", "next monitoring visit, 32 recipients"),
]
Y_HEAD = 84.0
for x, h in zip(COLX, HEAD):
    axB.text(x, Y_HEAD, h, fontsize=8.2, fontweight="bold", color=INK, va="center")
axB.plot([0.8, 99.2], [Y_HEAD - 6.0, Y_HEAD - 6.0], color=INK, lw=0.9)
ROW_H = 15.0
for i, (key, q, data) in enumerate(ROWS):
    col = MODEL[key]
    yc = Y_HEAD - 6.0 - ROW_H * (i + 0.5)
    axB.text(2.4, yc, str(i + 1), fontsize=9.5, fontweight="bold", color=col, ha="center", va="center")
    axB.text(COLX[0], yc, q, fontsize=8.2, color=INK, va="center")
    axB.text(COLX[1], yc, data, fontsize=8.2, color=col, va="center")
    if i < len(ROWS) - 1:
        axB.plot([0.8, 99.2], [yc - ROW_H / 2, yc - ROW_H / 2], color="#d6d6d6", lw=0.6)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
from PIL import Image, ImageOps                              # trim the white margin left by the invisible axes
im = Image.open(OUT).convert("RGB"); bb = ImageOps.invert(im).getbbox()
pad = 12; im.crop((max(bb[0] - pad, 0), max(bb[1] - pad, 0), min(bb[2] + pad, im.size[0]), min(bb[3] + pad, im.size[1]))).save(OUT, dpi=(300, 300))
print("wrote", OUT)
