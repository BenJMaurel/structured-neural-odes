#!/usr/bin/env python3
"""Fig. 2 of ndm.tex: where the dose acts, four designs in the order of the paper.
  (a) plain latent ODE (step 1): the dose only reaches the encoder; trained on single visits it stays inert
  (b) dose-conditioned field: the dose enters the vector field
  (c) OT-FiLM (step 2): the dose transports the initial state, one dose-free field
  (d) dose-event model (step 3): each dose jumps the drug coordinates, the physiology coordinates are conserved
Panels (b) and (c) redraw figures/dose_two_ways.png with the code of
presentation/figure_scripts/make_dose_two_ways_dynamics.py (paper vocabulary). Schematic, not model output.

  python3 make_fig_designs.py [outfile]   (default figures/dose_designs_four.png)
"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../../..'))
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/figures/dose_designs_four.png"
RED, BLUE, TEAL, GREY, INK, PURPLE, SLATE = "#c8313a", "#1f5673", "#12807b", "#a8a8a8", "#111111", "#6a3d9a", "#555555"
XL, YL = (0.0, 10.0), (0.0, 8.0)


def unit(phi):
    return np.cos(phi), np.sin(phi)


def phi_dc(x, y, dose):          # (b) one direction field per dose; both curved: linear fields were misleading)
    if dose == 2:
        return -1.00 + 0.19 * x + 0.12 * np.sin(0.7 * y)
    return 0.42 - 0.17 * x + 0.12 * np.sin(0.7 * y)


def phi_one(x, y):               # (a), (c) one field that never sees the dose
    return -0.36 + 0.095 * x


def integrate(phi, start, xmax, step=0.02, nmax=4000):
    p = np.array(start, float); out = [p.copy()]
    for _ in range(nmax):
        dx, dy = unit(phi(p[0], p[1]))
        p = p + step * np.array([dx, dy]); out.append(p.copy())
        if p[0] >= xmax or not (YL[0] < p[1] < YL[1]):
            break
    return np.array(out)


def frame(ax):
    ax.set_xlim(*XL); ax.set_ylim(*YL)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(INK); s.set_linewidth(1.0)


def arrows(ax, phi, color, alpha, width=0.0046):
    gx, gy = np.meshgrid(np.linspace(1.0, 9.2, 10), np.linspace(1.3, 6.1, 7))
    u, v = unit(phi(gx, gy))
    ax.quiver(gx, gy, u, v, color=color, alpha=alpha, angles="xy", scale_units="xy", scale=1 / 0.52, width=width,
              headwidth=4.6, headlength=5.0, pivot="mid", zorder=1)


def legend(ax, labels=("2 mg", "8 mg"), title="dose", loc="upper right"):
    h = [Line2D([], [], color=BLUE, lw=4, label=labels[0]), Line2D([], [], color=RED, lw=4, label=labels[1])]
    ax.legend(handles=h, title=title, loc=loc, fontsize=11.5, title_fontsize=11.5, frameon=True, framealpha=0.96)


def caption(ax, text, color):
    ax.text(0.5, 0.045, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=12.6, fontweight="bold",
            color=color, zorder=10, bbox=dict(boxstyle="square,pad=0.35", fc="white", ec=color, lw=1.5))


def equation(ax, text, color, x=0.5):
    ax.text(x, 0.93, text, transform=ax.transAxes, ha="center", va="top", fontsize=14, color=color)


fig, axs = plt.subplots(2, 2, figsize=(11.4, 9.3))
fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, wspace=0.03, hspace=0.13)
axA, axB, axC, axD = axs.ravel()

# ---------------------------------------------------------------- (a) plain latent ODE
frame(axA)
arrows(axA, phi_one, GREY, 0.95)
s2, s8 = (1.3, 3.15), (1.3, 3.32)          # the encoder barely moves z0 when only its dose input changes
for s, col, ls in ((s2, BLUE, "-"), (s8, RED, (0, (3, 2)))):
    tr = integrate(phi_one, s, xmax=8.9)
    axA.plot(tr[:, 0], tr[:, 1], color=col, lw=4.0, ls=ls, solid_capstyle="round", zorder=4)
    axA.plot(*tr[-1], "o", color=col, ms=10, zorder=5)
axA.plot(*s2, "o", color=BLUE, ms=14, zorder=6); axA.plot(*s8, "o", color=RED, ms=9, zorder=7)
axA.text(s2[0] + 0.3, s2[1] + 0.55, "encoder output", fontsize=12, color=INK, zorder=6)
equation(axA, r"$\dot z = f(z,\,t)$,   $z_0 = \mathrm{enc}(\mathrm{samples},\,d)$", SLATE, x=0.42)
legend(axA)
caption(axA, "trained on single visits: changing d barely moves z0\n⇒ the same exposure at every dose", SLATE)
axA.set_title("(a)  plain latent ODE: the dose stays inert", fontsize=14.5, color=SLATE, pad=7)

# ---------------------------------------------------------------- (b) dose-conditioned field
frame(axB)
arrows(axB, lambda x, y: phi_dc(x, y, 2), BLUE, 0.60)
arrows(axB, lambda x, y: phi_dc(x, y, 8), RED, 0.60)
start = (1.3, 6.35)
for dose, col in ((2, BLUE), (8, RED)):
    tr = integrate(lambda x, y: phi_dc(x, y, dose), start, xmax=8.6, nmax=600)
    axB.plot(tr[:, 0], tr[:, 1], color=col, lw=4.2, solid_capstyle="round", zorder=4)
    axB.plot(*tr[-1], "o", color=col, ms=11, zorder=5)
axB.plot(*start, "o", color=INK, ms=15, zorder=6)
axB.text(start[0] + 0.35, start[1] + 0.42, "one starting point", fontsize=12.5, color=INK, zorder=6)
equation(axB, r"$\dot z = f(z,\,t,\,\mathbf{d})$", RED, x=0.45)
legend(axB)
caption(axB, "two doses  ⇒  two sets of dynamics\nthe dose rewrites the rules of motion", RED)
axB.set_title("(b)  dose in the dynamics", fontsize=14.5, color=RED, pad=7)

# ---------------------------------------------------------------- (c) OT-FiLM
frame(axC)
arrows(axC, phi_one, GREY, 0.95)
s_lo, s_hi = (1.3, 2.05), (1.3, 4.35)
ends = {}
for s, col in ((s_lo, BLUE), (s_hi, RED)):
    tr = integrate(phi_one, s, xmax=8.9)
    axC.plot(tr[:, 0], tr[:, 1], color=col, lw=4.2, solid_capstyle="round", zorder=4)
    axC.plot(*s, "o", color=col, ms=14, zorder=6)
    axC.plot(*tr[-1], "o", color=col, ms=11, zorder=5)
    ends[col] = tr[-1]
axC.annotate("", xy=(s_hi[0], s_hi[1] - 0.32), xytext=(s_lo[0], s_lo[1] + 0.32),
             arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=3.4, mutation_scale=22), zorder=7)
axC.text(s_lo[0] - 0.55, (s_lo[1] + s_hi[1]) / 2, "dose shift", rotation=90, ha="center", va="center", fontsize=13,
         fontweight="bold", color=TEAL)
eb, er = ends[BLUE], ends[RED]
axC.annotate("", xy=(eb[0], eb[1] + 0.25), xytext=(er[0], er[1] - 0.25),
             arrowprops=dict(arrowstyle="<|-|>", ls=(0, (2, 2)), lw=2.0, color=INK), zorder=6)
axC.text(eb[0] - 0.25, (eb[1] + er[1]) / 2, "different\nexposure", ha="right", va="center", fontsize=12,
         fontweight="bold", color=INK, zorder=8, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
equation(axC, r"$z_0 \mapsto \gamma(d,d')\odot z_0+\beta(d,d')$,   $\dot z = f(z,\,t)$", TEAL, x=0.38)
legend(axC, loc="upper right")
caption(axC, "one set of dynamics  ⇒  same rules\nthe dose only chooses where you start", TEAL)
axC.set_title("(c)  dose-shift model: the dose moves the initial state", fontsize=14.5, color=TEAL, pad=7)

# ---------------------------------------------------------------- (d) dose-event model: the latent space
# Two drug coordinates (u1 ~ depot, u2 ~ central) horizontally, physiology p vertically. A patient lives on the sheet p = const;
# each dose jumps the drug state along b(p) inside that sheet; the drug flow du/dt = A(p) u brings it back to the drug-free axis
# u = 0, shared by all patients. Schematic (a linear 2-compartment flow), not model output.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
axF = axD                                                  # 2-D frame, title, legend and caption, as panels (a)-(c)
frame(axF)
pos = axF.get_position()
axD = fig.add_axes([pos.x0 + 0.005, pos.y0 + 0.075, pos.width - 0.01, pos.height - 0.085], projection="3d")
axD.set_box_aspect((1.35, 1.0, 0.95), zoom=1.22)
axD.set_zorder(0); axD.patch.set_alpha(0.0)
axF.set_zorder(1); axF.patch.set_alpha(0.0)          # frame, legend and caption above the 3-D view
axD.set_proj_type("ortho"); axD.view_init(elev=19, azim=-58)


def flow_path(p, times, amts, t_end=24.0, dt=0.02):
    ka, ke = 0.55, 0.10 + 0.30 * (1 - p)               # physiology sets the elimination (and the jump below)
    A = np.array([[-ka, 0.0], [ka, -ke]])
    b = np.array([np.cos(0.05 + 0.75 * p), np.sin(0.05 + 0.75 * p)])  # b(p): the dose direction depends on the patient
    u = np.zeros(2); out, jumps = [u.copy()], []
    for t in np.arange(0, t_end + dt, dt):
        for tk, a in zip(times, amts):
            if abs(t - tk) < dt / 2:
                new = u + 0.25 * a * b; jumps.append((u.copy(), new.copy())); u = new
                out.append(np.array([np.nan, np.nan])); out.append(u.copy())   # the jump is drawn separately
        u = u + dt * (A @ u); out.append(u.copy())
    return np.array(out), jumps


SHEETS = {0.25: "patient 1", 0.80: "patient 2"}
for p_, lab in SHEETS.items():                              # translucent sheet of each patient
    X, Y = np.meshgrid([-0.05, 1.25], [-0.05, 0.95])
    axD.plot_surface(X, Y, np.full_like(X, p_), color=PURPLE, alpha=0.07, edgecolor="none", zorder=0)
    axD.text(1.25, 0.95, p_ + 0.17, f"{lab}: $p={'p_' + lab[-1]}$", color=PURPLE, fontsize=10.5, ha="right", va="bottom",
             fontweight="bold")
    gx, gy = np.meshgrid(np.linspace(0.12, 1.15, 5), np.linspace(0.08, 0.85, 4))   # the flow on the sheet
    ka, ke = 0.55, 0.10 + 0.30 * (1 - p_)
    du, dv = -ka * gx, ka * gx - ke * gy
    nrm = np.hypot(du, dv) + 1e-9
    axD.quiver(gx, gy, np.full_like(gx, p_), 0.09 * du / nrm, 0.09 * dv / nrm, 0 * gx, color=GREY, alpha=0.9,
               linewidth=0.9, arrow_length_ratio=0.45)
axD.plot([0, 0], [0, 0], [0.0, 1.05], color=INK, lw=2.4, zorder=5)    # drug-free axis
axD.text(0.0, 0.0, -0.13, "no drug\n($u=0$)", color=INK, fontsize=10.5, ha="center", va="top")

REG = {BLUE: ([0], [2]), RED: ([0, 3], [4, 4])}
for p_, cols in ((0.25, (BLUE, RED)), (0.80, (BLUE,))):
    for col in cols:
        tr, jumps = flow_path(p_, *REG[col])
        axD.plot(tr[:, 0], tr[:, 1], np.full(len(tr), p_), color=col, lw=2.4 if p_ < 0.5 else 1.9, zorder=6)
        for u0, u1 in jumps:
            axD.quiver(u0[0], u0[1], p_, u1[0] - u0[0], u1[1] - u0[1], 0, color=col, linewidth=1.6,
                       arrow_length_ratio=0.12, linestyle=(0, (2, 1.2)), zorder=7)
axD.set_xlim(-0.05, 1.25); axD.set_ylim(-0.05, 0.95); axD.set_zlim(0, 1.05)
axD.set_xticks([]); axD.set_yticks([]); axD.set_zticks([])
axD.set_xlabel("drug $u_1$", fontsize=11, labelpad=-12, color=PURPLE)
axD.set_ylabel("drug $u_2$", fontsize=11, labelpad=-12, color=PURPLE)
axD.set_zlabel("physiology $p$", fontsize=11, labelpad=-12, color=PURPLE)
for a in (axD.xaxis, axD.yaxis, axD.zaxis):
    a.pane.set_facecolor((1, 1, 1, 0)); a.pane.set_edgecolor("#cccccc")
axD.grid(False)
h = [Line2D([], [], color=BLUE, lw=3, label="2 mg once"), Line2D([], [], color=RED, lw=3, label="4 mg at 0 and 3 h"),
     Line2D([], [], color=INK, lw=1.6, ls=(0, (2, 1.2)), label="dose = jump of $u$")]
axF.legend(handles=h, title="regimen", loc="upper left", fontsize=10.5, title_fontsize=10.5, frameon=True, framealpha=0.96)
axF.text(0.5, 0.045, "a patient stays on its sheet $p$; a dose adds to $u$,\nthe flow brings $u$ back to the drug-free axis",
           transform=axF.transAxes, ha="center", va="bottom", fontsize=12, fontweight="bold", color=PURPLE, zorder=10,
           bbox=dict(boxstyle="square,pad=0.35", fc="white", ec=PURPLE, lw=1.5))
axF.set_title("(d)  dose-event model", fontsize=14.5, color=PURPLE, pad=7)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor="white")
print("wrote", OUT)
