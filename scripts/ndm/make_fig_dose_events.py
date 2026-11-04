#!/usr/bin/env python3
"""How each model handles a dosing history (several doses), in the style of Fig. 2.; then "the 3D figures are really confusing ... find a way to concile" ->
TIME-STRIP version (3-D version kept as make_fig_dose_events.py.bak_3d).
Each panel: one patient under two regimens (blue A: 2 mg once; red B: 4 mg at 0 and 3 h), two aligned strips against time:
  top    = what the dose acts on (a compartment, a latent coordinate, a pulse through the field, a trace)
  bottom = the coordinate that identifies the patient. A patient conserved by construction is ONE flat line shared by both
           regimens (tinted strip); confusion of the physiology shows as lines that start apart, move with the doses or separate.
Where the dose enters (chips: encoder / prior / field / state jump) and the equations are read from the code we ran:
  (a) truth: pharmacokinetic model; (b) Lu et al. iScience (lu_regimen.py 'is'); (c) Lu et al. NMI (lu_regimen.py 'nmi');
  (d) EB-VAE (third_party/baaz_ebvae/lib/models/NNmodels.py); (e) dose-conditioned field (dh_regimen.py --dose-cond);
  (f) dose-event model (dh_regimen.py, Table 4 architecture); (g) summary table. Curves are SCHEMATIC toy dynamics, not model output.

  python3 make_fig_dose_events.py [outfile]   (default figures/dose_events_models.png, + .pdf)
"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/figures/dose_events_models.png"
RED, BLUE, GREY, INK, PURPLE, SLATE = "#c8313a", "#1f5673", "#a8a8a8", "#111111", "#6a3d9a", "#555555"
GOOD, BAD = "#1a7f37", "#b42318"
TINT = "#f4f0fa"
DT = 0.01
T = np.arange(0.0, 16.0 + DT / 2, DT)
REG = {"A": (BLUE, [0.0], [2.0]), "B": (RED, [0.0, 3.0], [4.0, 4.0])}


def kicks(times, amts):
    """amount added at each grid step"""
    k = np.zeros_like(T)
    for tk, a in zip(times, amts):
        k[int(round(tk / DT))] += a
    return k


def jump_decay(times, amts, rate):
    """a coordinate that jumps by the amount at each dose and relaxes at `rate` (the flow)"""
    k, x, out = kicks(times, amts), 0.0, np.zeros_like(T)
    for i in range(len(T)):
        x += k[i]; out[i] = x; x -= DT * rate * x
    return out


def depot_central(times, amts, ka=0.9, ke=0.22):
    """truth: total drug amount in the body (depot + central), the dose added to the depot"""
    k, d, c, out = kicks(times, amts), 0.0, 0.0, np.zeros_like(T)
    for i in range(len(T)):
        d += k[i]; out[i] = d + c
        d, c = d - DT * ka * d, c + DT * (ka * d - ke * c)
    return out


def pulse_drug(times, amts, rate=0.45, sigma=0.6, beta=0.11, gain=0.95):
    """EB-VAE: half-Gaussian pulse v(t), passed through a concave psi(v) = v - beta v^2, drives the drug coordinate"""
    v = sum(a * np.exp(-((T - tk) / sigma) ** 2) * (T >= tk) for tk, a in zip(times, amts))
    x, out = 0.0, np.zeros_like(T)
    for i in range(len(T)):
        out[i] = x; x += DT * (gain * (v[i] - beta * v[i] ** 2) - rate * x)
    return out, v


def trace(times, amts, tau=1.5):
    return sum(a * np.exp(-(T - tk) / tau) * (T >= tk) for tk, a in zip(times, amts))


def drift(start, signal, c):
    return start + c * np.cumsum(signal) * DT


def frame(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(INK); s.set_linewidth(1.0)


def chips(ax, used, y=0.94):
    ax.text(0.03, y, "the dose enters:", transform=ax.transAxes, fontsize=11.5, color=SLATE, va="center", ha="left")
    for n, x in (("encoder", 0.255), ("prior", 0.405), ("field", 0.515), ("state jump", 0.625)):
        on = n in used
        ax.text(x, y, n, transform=ax.transAxes, fontsize=11.0, va="center", ha="left", fontweight="bold" if on else "normal",
                color="white" if on else "#9a9a9a",
                bbox=dict(boxstyle="round,pad=0.28", fc=INK if on else "white", ec=INK if on else "#c8c8c8", lw=1.0))


def equation(ax, lines, color=INK, y=0.885):
    ax.text(0.5, y, "\n".join(lines), transform=ax.transAxes, ha="center", va="top", fontsize=12.0, color=color, linespacing=1.45)


def caption(ax, text, color):
    ax.text(0.5, 0.018, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=11.2, fontweight="bold", color=color,
            zorder=10, linespacing=1.3, bbox=dict(boxstyle="square,pad=0.35", fc="white", ec=color, lw=1.4))


def strips(fig, axF, top_label, bot_label, fixed, bot_color=SLATE):
    """two aligned time strips inside the panel frame axF: top = what the dose acts on, bottom = the patient coordinate"""
    pos = axF.get_position()
    L, W = pos.x0 + 0.105 * pos.width, 0.80 * pos.width
    top = fig.add_axes([L, pos.y0 + 0.495 * pos.height, W, 0.195 * pos.height])
    bot = fig.add_axes([L, pos.y0 + 0.255 * pos.height, W, 0.195 * pos.height], sharex=top)
    for a in (top, bot):
        a.set_xlim(T[0], T[-1]); a.tick_params(labelsize=9.5)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    top.tick_params(labelbottom=False); top.set_yticks([])
    bot.set_xticks([0, 3, 6, 9, 12, 15]); bot.set_xlabel("time (h)", fontsize=10.5, labelpad=1); bot.set_yticks([])
    top.set_ylabel(top_label, fontsize=10.5, color=SLATE, labelpad=6)
    bot.set_ylabel(bot_label, fontsize=10.5, color=bot_color, labelpad=6)
    bot.set_ylim(0.25, 0.92)
    if fixed:
        bot.set_facecolor(TINT)
        bot.text(0.995, 0.93, "fixed by construction", transform=bot.transAxes, ha="right", va="top", fontsize=10,
                 color=PURPLE, fontweight="bold")
    else:
        bot.text(0.995, 0.93, "not fixed by construction", transform=bot.transAxes, ha="right", va="top", fontsize=10,
                 color=BAD, fontweight="bold")
    return top, bot


def dose_marks(ax):
    """triangles under the top strip at each dose, coloured by regimen (A slightly left of B at t = 0)"""
    for key, off in (("A", -0.12), ("B", 0.12)):
        col, times, amts = REG[key]
        for tk, a in zip(times, amts):
            ax.plot(tk + off, 0.0, marker="^", color=col, ms=8 + 1.5 * a, transform=ax.get_xaxis_transform(), clip_on=False,
                    zorder=6)


def same_line(ax, y):
    """the two regimens give the same patient coordinate: blue solid with red dashed on top"""
    ax.plot(T, y, color=BLUE, lw=4.0, solid_capstyle="round")
    ax.plot(T, y, color=RED, lw=2.0, ls=(0, (4, 3)))


def note(ax, x, y, text, color=INK, ha="left", va="center", size=10.2):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, color=color, ha=ha, va=va, linespacing=1.2, zorder=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


fig = plt.figure(figsize=(14.2, 22.0))
gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 0.74], left=0.012, right=0.988, top=0.926, bottom=0.006, wspace=0.03,
                      hspace=0.085)
axs = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(2)]
for a in axs:
    frame(a)

hl = [Line2D([], [], color=BLUE, lw=3.2, marker="^", ms=8, label="regimen A: 2 mg once"),
      Line2D([], [], color=RED, lw=3.2, marker="^", ms=10, label="regimen B: 4 mg at 0 and 3 h"),
      Line2D([], [], color=RED, lw=2.0, ls=(0, (4, 3)), label="red dashed on blue: both regimens give the same line"),
      Patch(fc=TINT, ec=PURPLE, lw=0.6, label="patient coordinate fixed by construction")]
fig.legend(handles=hl, loc="upper center", ncol=2, fontsize=11.5, frameon=False, bbox_to_anchor=(0.5, 0.999))
fig.text(0.5, 0.950, "One patient, two regimens, against time. Top strip: what the dose acts on.  Bottom strip: the coordinate "
         "that identifies the patient.\nSchematic curves (not model output); where the dose enters and the equations are those "
         "of the code we ran.", ha="center", va="center", fontsize=11, color=SLATE, style="italic")

# ---------------------------------------------------------------- (a) truth
ax = axs[0]
chips(ax, {"state jump"})
equation(ax, [r"at each dose $x \mapsto x + a\,e_{\mathrm{depot}}$,    $\dot x = g(x;\theta)$,    $\dot\theta = 0$",
              r"$C = x_{\mathrm{central}}\,/\,V(\theta)$"], SLATE)
top, bot = strips(fig, ax, "drug amount", r"parameters $\theta$", True)
for key in ("A", "B"):
    col, times, amts = REG[key]
    top.plot(T, depot_central(times, amts), color=col, lw=2.4)
top.axhline(0, color=INK, lw=0.8, ls=":"); dose_marks(top)
same_line(bot, np.full_like(T, 0.55))
note(bot, 0.30, 0.30, "the same θ under both regimens")
caption(ax, "a dose adds drug to the depot; the parameters θ never change;\nwith no drug left, every patient returns to zero",
        SLATE)
ax.set_title("(a)  the truth: a pharmacokinetic model", fontsize=15, color=SLATE, pad=7)

# ---------------------------------------------------------------- (b) Lu et al. iScience
ax = axs[1]
chips(ax, {"encoder", "state jump"})
equation(ax, [r"$z_0 = \mathrm{GRU}(\mathrm{observed\ PK},\ \mathbf{regimen\ to\ predict})$,    at each dose $z_1 \mapsto z_1 + a$",
              r"$\dot z = f(z)$  (unstructured),    $C = h(z,\ \mathrm{observed\ PK})$"])
top, bot = strips(fig, ax, r"dosed $z_1$", "patient\ndirection", False)
for key, z0 in (("A", 0.42), ("B", 0.56)):          # the encoder reads the regimen to predict: z0 moves with it
    col, times, amts = REG[key]
    z1 = jump_decay(times, amts, 0.35)
    top.plot(T, z1, color=col, lw=2.4)
    bot.plot(T, drift(z0, z1, 0.0075), color=col, lw=2.8)
dose_marks(top)
note(bot, 0.02, 0.09, "starts apart: the encoder reads the regimen to predict")
note(bot, 0.36, 0.89, "moves while drug is present", ha="center")
caption(ax, "the encoder reads the regimen to predict, and the field can\nmove every coordinate after a dose:\n"
            "the same patient becomes different patients", INK)
ax.set_title("(b)  Lu et al., iScience 2021", fontsize=15, color=INK, pad=7)

# ---------------------------------------------------------------- (c) Lu et al. NMI
ax = axs[2]
chips(ax, {"encoder", "state jump"})
equation(ax, [r"$\theta = \mathrm{GRU}(\mathrm{observed\ PK},\ \mathrm{observed\ amounts})$,    at each dose $s_1 \mapsto s_1 + a$",
              r"$\dot s = \mathrm{PKVF}(s,\theta)$,    $\dot\theta = 0$,    $C = s_2$"])
top, bot = strips(fig, ax, r"dosed $s_1$", r"parameters $\theta$", True)
for key in ("A", "B"):
    col, times, amts = REG[key]
    top.plot(T, jump_decay(times, amts, 0.9), color=col, lw=2.4)
dose_marks(top)
note(top, 0.62, 0.62, r"$s=0$ is not imposed as a rest point", ha="center")
same_line(bot, np.full_like(T, 0.55))
note(bot, 0.30, 0.30, "the same θ under both regimens")
caption(ax, "a neural PK model: θ fixed, the dose adds to a hand-chosen\ncompartment and C is the other one;\n"
            "but the encoder reads the amounts and no rest state is imposed", INK)
ax.set_title("(c)  Lu et al., Nature Machine Intelligence 2021", fontsize=15, color=INK, pad=7)

# ---------------------------------------------------------------- (d) EB-VAE
ax = axs[3]
chips(ax, {"prior", "field"})
equation(ax, [r"$\dot u = F(u,\,p,\,\psi(\sum_k a_k\,\phi(t-t_k)))$,    $\dot p = 0$,    $C = c^{\top}u$",
              r"$\phi$: half-Gaussian pulse,   $\psi$: small network,   prior mean of $p$: $\mu(\mathrm{covariates},\,a_1)$"])
top, bot = strips(fig, ax, r"drug $u_1$", r"parameters $p$", True)
rA, vA = pulse_drug(*REG["A"][1:])
for key in ("A", "B"):
    col, times, amts = REG[key]
    r, v = pulse_drug(times, amts)
    top.plot(T, r, color=col, lw=2.4)
prop = 2 * rA + 2 * np.interp(T - 3.0, T, rA, left=0.0)          # what a proportional response to B would be
top.plot(T, prop, color=GREY, lw=2.0, ls=(0, (2, 2)))
note(top, 0.64, 0.80, "grey dotted: B if the effect were\nproportional to the amount", color=SLATE, ha="center")
dose_marks(top)
same_line(bot, np.full_like(T, 0.55))
note(bot, 0.30, 0.17, r"the same $p$ under both regimens; its prior mean" "\n" r"depends on the first amount $a_1$")
caption(ax, "the patient p is fixed, but the dose is a pulse through a learned\nmap: its effect need not scale with the "
            "amount,\nand the prior reads the dose", INK)
ax.set_title("(d)  EB-VAE (Baaz et al.)", fontsize=15, color=INK, pad=7)

# ---------------------------------------------------------------- (e) dose-conditioned field with a dose trace
ax = axs[4]
chips(ax, {"field"})
equation(ax, [r"$\dot z = f(z,\,w(t))$  (unstructured),    $w(t) = \sum_{t_k\leq t} a_k\, e^{-(t-t_k)/\tau}$",
              r"$C = h(z)$;   encoder: concentrations / dose (no dose value)"])
top, bot = strips(fig, ax, r"trace $w(t)$", "patient\ndirection", False)
for key in ("A", "B"):
    col, times, amts = REG[key]
    w = trace(times, amts)
    top.plot(T, w, color=col, lw=2.4)
    bot.plot(T, drift(0.45, w, 0.012), color=col, lw=2.8)
dose_marks(top)
note(top, 0.62, 0.62, "the latent state does not jump:\nthe field changes while the trace is on", ha="center")
note(bot, 0.02, 0.89, "drifts in proportion to the cumulative dose")
caption(ax, "the dose is a decaying input to an unstructured field:\nwhile it is on, every coordinate can move,\n"
            "and the patient drifts with the cumulative dose", INK)
ax.set_title("(e)  dose-conditioned field", fontsize=15, color=INK, pad=7)

# ---------------------------------------------------------------- (f) dose-event model (ours)
ax = axs[5]
chips(ax, {"state jump"})
equation(ax, [r"$(u_0,p) = \mathrm{enc}(C/\mathrm{dose})$,    at each dose $u \mapsto u + a\,b(p)$",
              r"$\dot u = f(u,p) - f(0,p) - \kappa\odot u$,    $\dot p = 0$,    $C = h(u,p)$"], PURPLE)
top, bot = strips(fig, ax, r"drug state $u$", r"physiology $p$", True, PURPLE)
for key in ("A", "B"):
    col, times, amts = REG[key]
    top.plot(T, jump_decay(times, amts, 0.35), color=col, lw=2.4)
top.axhline(0, color=INK, lw=0.8, ls=":"); dose_marks(top)
note(top, 0.62, 0.62, r"rests at $u=0$ when no drug is left, for every $p$", color=PURPLE, ha="center")
same_line(bot, np.full_like(T, 0.55))
note(bot, 0.30, 0.30, r"the same $p$ under any regimen", color=PURPLE)
caption(ax, "a dose adds an amount to the drug state along b(p);\nthe physiology p never moves, so the same patient\n"
            "is predicted under any regimen", PURPLE)
ax.set_title("(f)  dose-event model (ours)", fontsize=15, color=PURPLE, pad=7)

# ---------------------------------------------------------------- (g) summary table
tab = fig.add_subplot(gs[3, :]); frame(tab)
tab.set_title("(g)  what each design builds in", fontsize=15, color=INK, pad=7)
COLS = ["the dose is an amount\nadded to the state", "the patient is conserved\nby construction",
        "no drug ⇒ rest state,\nby construction", "the patient estimate does\nnot read the dose value"]
ROWS = [("(a) truth (PK model)", SLATE, [("y", "into the depot"), ("y", "θ constant"), ("y", "x = 0"), ("-", "")]),
        ("(b) Lu et al., iScience", INK, [("y", "into z₁"), ("n", "unstructured field"), ("n", ""),
                                          ("n", "reads the regimen to predict")]),
        ("(c) Lu et al., NMI", INK, [("y", "into s₁"), ("y", "θ constant"), ("n", ""), ("n", "reads the observed amounts")]),
        ("(d) EB-VAE", INK, [("n", "pulse through ψ into the field"), ("y", "p constant"), ("n", ""),
                             ("n", "prior mean reads the first amount")]),
        ("(e) dose-conditioned field", INK, [("n", "decaying input to the field"), ("n", "unstructured field"), ("n", ""),
                                             ("y", "C / dose (as run here)")]),
        ("(f) dose-event model (ours)", PURPLE, [("y", "along b(p)"), ("y", "p constant"), ("y", "u = 0 for every p"),
                                                 ("y", "C / dose")])]
x0, cw = 0.235, (0.985 - 0.235) / 4
ytop, rh = 0.80, 0.122
for i, c in enumerate(COLS):
    tab.text(x0 + (i + 0.5) * cw, 0.905, c, ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK, linespacing=1.2)
tab.plot([0.015, 0.985], [0.835, 0.835], color=INK, lw=1.0)
MARK = {"y": ("✓", GOOD), "n": ("✗", BAD), "-": ("—", GREY)}
for r, (name, col, cells) in enumerate(ROWS):
    y = ytop - (r + 0.5) * rh
    if name.startswith("(f)"):
        tab.add_patch(FancyBboxPatch((0.012, y - rh / 2 + 0.008), 0.976, rh - 0.016, boxstyle="round,pad=0.004", fc=PURPLE,
                                     alpha=0.07, ec="none", transform=tab.transAxes))
    tab.text(0.025, y, name, ha="left", va="center", fontsize=12.5, color=col, fontweight="bold" if col == PURPLE else "normal")
    for i, (m, note) in enumerate(cells):
        sym, mc = MARK[m]
        tab.text(x0 + (i + 0.5) * cw, y + (0.018 if note else 0), sym, ha="center", va="center", fontsize=17, color=mc,
                 fontweight="bold")
        if note:
            tab.text(x0 + (i + 0.5) * cw, y - 0.035, note, ha="center", va="center", fontsize=10, color=SLATE)
    if r < len(ROWS) - 1:
        tab.plot([0.015, 0.985], [y - rh / 2, y - rh / 2], color="#e2e2e2", lw=0.8)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=170, facecolor="white")
fig.savefig(os.path.splitext(OUT)[0] + ".pdf", facecolor="white")
print("wrote", OUT)
