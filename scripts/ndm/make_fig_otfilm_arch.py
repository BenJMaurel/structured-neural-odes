#!/usr/bin/env python3
"""Architecture of OT-FiLM, the dose-transport model of step 2 (ndm.tex). Same visual language as make_fig_dose_event_arch.py (Fig. of step 3), replacing the
TikZ figures/Architecture.pdf (which still said "FiLM Modulation" / "Decoder" / "Base Model"). Drawn to match the code
(run_models.py --use_film --film-no-z0-cond --decoder-hidden 50 --decoder-residual; lib/latent_ode.py):
  encoder    ODE-RNN on the three samples (0, 1, 3 h) with the dose, covariates -> h0 -> Gaussian posterior over z0 (10 dims)
  transport  z0' = gamma(d, d') * z0 + beta(d, d'), gamma, beta functions of the dose pair only (W2 map between diagonal Gaussians)
  training   the transported posterior is matched to the posterior encoded from the other visit of the same patient (W2)
  field      dz/dt = f(z, t), never receives the dose;  readout residual h(z) = W z + b + g(z)
  step 1     the plain latent ODE is the same network without the transport (said in the caption, not drawn)
Writes figures/otfilm_architecture.{pdf,png} in the manuscript folder."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = f'{REPO}/figures/otfilm_architecture'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'mathtext.fontset': 'dejavusans'})

INK, MUTED = '#222222', '#555555'
ENC_E, ENC_F = '#3b6ea5', '#e8f0f9'          # inference network (as in the step-3 figure)
TR_E, TR_F = '#12807b', '#e3f3f1'            # dose transport (step-2 colour of Fig. 1)
OUT_E, OUT_F = '#2e7d4f', '#e9f5ee'          # field, readout, prediction
DOSE = '#d9822b'                             # doses (as in the step-3 figure)
BLUE = '#1f5673'

fig = plt.figure(figsize=(5.4, 3.75))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis('off')


def box(x, y, w, h, title, body=None, ec='#777777', fc='#f2f2f2', tsize=7.0, bsize=6.4, lw=1.0, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', ec=ec, fc=fc, lw=lw, zorder=2))
    if body is None:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=tsize, color=INK, fontweight='bold', zorder=3)
    else:
        ax.text(x + w / 2, y + h - 1.3, title, ha='center', va='top', fontsize=tsize, color=INK, fontweight='bold', zorder=3)
        ax.text(x + w / 2, y + h - 4.6, body, ha='center', va='top', fontsize=bsize, color=INK, zorder=3, linespacing=1.35)


def arrow(p0, p1, color=INK, lw=1.1, ls='-', rad=0.0, ms=8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', zorder=4, shrinkA=0, shrinkB=0))


def panel(x, y, w, h, label, ec, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=2.0', ec=ec, fc=fc, lw=0.9, alpha=0.55, zorder=0))
    ax.text(x + w - 1.2, y + h - 1.0, label, ha='right', va='top', fontsize=7.4, color=ec, fontweight='bold', zorder=1)


# ---------------------------------------------------------------- inference network (top)
panel(1, 44.0, 98, 21.0, 'Inference network', ENC_E, '#f5f8fc')
box(3, 52.0, 22, 10.0, 'Observed visit', '3 samples (0, 1, 3 h)\nand its dose $d$', ENC_E, '#ffffff')
box(3, 45.5, 22, 5.5, 'Covariates', None, ENC_E, '#ffffff')
box(29.5, 52.0, 17, 10.0, 'ODE-RNN', 'encoder', ENC_E, ENC_F)
box(29.5, 45.5, 17, 5.5, 'MLP → $h_0$', None, ENC_E, ENC_F)
box(51, 49.0, 21, 12.0, 'Posterior', r'$q(z_0\mid \mathcal{O},s)$' + '\nGaussian, 10 dims', ENC_E, ENC_F)
arrow((25, 57.0), (29.5, 57.0)); arrow((25, 48.25), (29.5, 48.25)); arrow((38, 51.0), (38, 52.0))
arrow((46.5, 57.0), (51, 57.0))

# ---------------------------------------------------------------- dose transport (middle)
panel(1, 25.0, 98, 17.0, 'Dose transport', TR_E, '#f3faf9')
box(3, 27.5, 17, 9.5, 'Alternative dose', r"$d'$ (any, 1–8 mg)", DOSE, '#ffffff', tsize=6.3)
box(26, 27.0, 33, 11.5, 'Optimal-transport map',
    r"$z_0' = \gamma(d,d')\odot z_0 + \beta(d,d')$" + '\n' + r'$\gamma,\beta$: functions of the dose pair only', TR_E, TR_F)
arrow((20, 32.25), (26, 32.25), color=DOSE, lw=1.2)
arrow((61.5, 49.0), (49.0, 38.5), color=TR_E, lw=1.1, rad=0.15)
ax.text(57.6, 42.6, r'$z_0$', fontsize=7, color=INK, ha='left', va='center')
box(63, 27.0, 34, 11.5, 'Training on pairs of visits',
    'transported posterior matched to the\nposterior encoded from the other visit\nof the same patient (Wasserstein-2)', TR_E, '#ffffff', bsize=6.0)

# ---------------------------------------------------------------- generative model (bottom)
panel(1, 1, 98, 22.0, 'Generative model', OUT_E, '#f6fbf8')
box(3, 4.0, 33, 13.5, 'Dose-independent vector field', r'$\dot z = f_\phi(z,t)$', OUT_E, OUT_F)   # 21/10 a co-author's labels
box(41, 4.0, 17, 13.5, 'Readout', r'$C=h(z)$' + '\nresidual', OUT_E, OUT_F)
arrow((36, 10.75), (41, 10.75)); arrow((58, 10.75), (61.5, 10.75))
arrow((26.0, 27.0), (19.5, 17.5), color=TR_E, lw=1.1, rad=-0.1)
ax.text(24.6, 21.4, r"$z_0'$", fontsize=7, color=TR_E, ha='left', va='center')

# prediction inset: AUC at the observed dose and at the new dose, same patient
ia = fig.add_axes([0.655, 0.075, 0.27, 0.195])
t = np.linspace(0, 12, 400)
curve = lambda s: s * (np.exp(-0.18 * t) - np.exp(-1.6 * t))
ia.fill_between(t, curve(1.0), color=DOSE, alpha=0.15, lw=0)
ia.plot(t, curve(1.0), color=INK, lw=1.1, label=r'observed dose $d$')
ia.plot(t, curve(1.45), color=BLUE, lw=1.1, ls=(0, (3, 1.5)), label=r"alternative dose $d'$")
ia.set_xticks([]); ia.set_yticks([]); ia.set_xlim(0, 12)
for s_ in ('top', 'right'):
    ia.spines[s_].set_visible(False)
ia.patch.set_alpha(0)
ia.legend(fontsize=5.6, frameon=False, loc='upper right', handlelength=2.0, bbox_to_anchor=(1.06, 1.0))
ia.set_xlabel('patient-specific prediction under alternative doses', fontsize=5.8, labelpad=2)

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}.{ext}', dpi=300, facecolor='white')
print(f'wrote {OUT}.{{pdf,png}}')
