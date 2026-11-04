#!/usr/bin/env python3
"""Architecture of the dose-event model (ndm.tex, step 3). Drawn to match the code (scripts/regimen/dh_regimen.py --kick-phys 32 --enc-dose norm; lib/encoder_decoder.py):
  encoder  ODE-RNN on the samples divided by the dose + number of doses before each sample (dose value withheld), covariates -> h0;
           on the two-visit task (two doses per training patient) the encoder also reads the dose (run_models.py --kick-phys 32,
           no --encoder-dose-norm; fallback (a))
  split    learned orthonormal Q: u = Q_d^T z (6 drug coordinates), p = Q_p^T z (4 physiology coordinates)
  jump     u <- u + (a_k / a_ref) b(p),  b(p) = b0 + g(p), g zero-initialised
  flow     du/dt = f(u, p) - f(0, p) - kappa * u,  dp/dt = 0
  readout  residual h(z) = W z + b + g(z) on the full latent state
Writes figures/dose_event_architecture.{pdf,png} in the manuscript folder."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = f'{REPO}/figures/dose_event_architecture'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'mathtext.fontset': 'dejavusans'})

INK, MUTED = '#222222', '#555555'
ENC_E, ENC_F = '#3b6ea5', '#e8f0f9'          # inference network
DRUG_E, DRUG_F = '#d9822b', '#fdf0e2'        # drug coordinates (dose colour of Fig. 1)
PHYS_E, PHYS_F = '#6b3fa0', '#f1eafa'        # physiology coordinates (step-3 colour of Fig. 1)
OUT_E, OUT_F = '#2e7d4f', '#e9f5ee'          # readout / prediction
GREY_E, GREY_F = '#777777', '#f2f2f2'

fig = plt.figure(figsize=(5.4, 3.75))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 66); ax.axis('off')


def box(x, y, w, h, title, body=None, ec=GREY_E, fc=GREY_F, tsize=7.0, bsize=6.4, lw=1.0, r=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', ec=ec, fc=fc, lw=lw, zorder=2))
    if body is None:
        ax.text(x + w / 2, y + h / 2, title, ha='center', va='center', fontsize=tsize, color=INK, fontweight='bold', zorder=3)
    else:
        ax.text(x + w / 2, y + h - 1.3, title, ha='center', va='top', fontsize=tsize, color=INK, fontweight='bold', zorder=3)
        ax.text(x + w / 2, y + h - 4.6, body, ha='center', va='top', fontsize=bsize, color=INK, zorder=3, linespacing=1.35)


def arrow(p0, p1, color=INK, lw=1.1, style='-|>', ls='-', rad=0.0, ms=8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 connectionstyle=f'arc3,rad={rad}', zorder=4, shrinkA=0, shrinkB=0))


def line(xs, ys, color, lw=1.2):
    ax.plot(xs, ys, color=color, lw=lw, zorder=4, solid_capstyle='round')


def panel(x, y, w, h, label, ec, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=2.0', ec=ec, fc=fc, lw=0.9, alpha=0.55, zorder=0))
    ax.text(x + w - 1.2, y + h - 1.0, label, ha='right', va='top', fontsize=7.4, color=ec, fontweight='bold', zorder=1)


# ---------------------------------------------------------------- inference network (top)
panel(1, 42.0, 98, 23.0, 'Inference network', ENC_E, '#f5f8fc')
box(3, 51.0, 22, 11.0, 'Observed samples', 'concentrations ÷ dose\n(or with the dose),\ndoses given so far', ENC_E, '#ffffff')
box(3, 43.5, 22, 6.0, 'Covariates', None, ENC_E, '#ffffff')
box(29.5, 51.0, 17, 11.0, 'ODE-RNN', 'encoder', ENC_E, ENC_F)
box(29.5, 43.5, 17, 6.0, 'MLP → $h_0$', None, ENC_E, ENC_F)
box(51, 49.5, 19, 12.5, 'Posterior', r'$q(z_0\mid \mathcal{O},s)$' + '\nGaussian, 10 dims', ENC_E, ENC_F)
box(74, 49.5, 23, 12.5, 'Learned split', r'$u=Q_{d}^{\top}z$ (6 drug)' + '\n' + r'$p=Q_{p}^{\top}z$ (4 physiol.)', GREY_E, '#ffffff')
arrow((25, 56.5), (29.5, 56.5)); arrow((25, 46.5), (29.5, 46.5)); arrow((38, 49.5), (38, 51.0))
arrow((46.5, 56.5), (51, 56.5)); arrow((70, 55.75), (74, 55.75))
ax.text(72.0, 56.5, r'$z_0$', ha='center', va='bottom', fontsize=7, color=INK)

# ---------------------------------------------------------------- generative model (bottom)
panel(1, 1, 98, 39.5, 'Generative model', OUT_E, '#f6fbf8')
# regimen (top left) -> dose event
box(3, 30.5, 24, 8.0, 'Regimen', r'any $(t_k,a_k)$', DRUG_E, '#ffffff')
arrow((10.5, 30.5), (10.5, 29.0), color=DRUG_E, lw=1.2)
# drug lane
box(3, 15.5, 21, 13.5, 'Dose event', r'at each $t_k$:' + '\n' + r'$u \leftarrow u + \frac{a_k}{a_{ref}}\, b(p)$', DRUG_E, DRUG_F)
box(27, 15.5, 30, 13.5, 'Drug flow', r'$\dot u = f_\phi(u,p)-f_\phi(0,p)-\kappa\odot u$' + '\n' + r'no drug stays no drug', DRUG_E, DRUG_F)
arrow((24, 22.25), (27, 22.25), color=DRUG_E, lw=1.2)
arrow((42, 29.0), (19.0, 29.0), color=DRUG_E, lw=0.9, rad=0.55, ls=(0, (3, 2)))
ax.text(38.5, 37.6, 'next dose', ha='center', va='center', fontsize=6.2, color=DRUG_E)
# initial drug state from the encoder
arrow((79, 49.5), (52.0, 29.0), color=DRUG_E, lw=1.1, rad=-0.05)
ax.text(70.0, 34.3, r'$u(t_0)=u_0$ or 0', ha='center', va='center', fontsize=6.2, color=DRUG_E)
# readout + prediction
box(60, 15.5, 17, 13.5, 'Readout', r'$C=h(z)$' + '\n' + 'residual', OUT_E, OUT_F)
arrow((57, 22.25), (60, 22.25), color=INK)
arrow((77, 22.25), (79.5, 22.25), color=INK)
# physiology lane
box(3, 3.0, 54, 8.5, 'Physiology coordinates  p', r'$\dot p = 0$: no dose acts on them', PHYS_E, PHYS_F)
arrow((13.5, 11.5), (13.5, 15.5), color=PHYS_E, lw=1.2); ax.text(14.2, 13.4, r'$b(p)$', fontsize=6.6, color=PHYS_E, ha='left', va='center')
arrow((42, 11.5), (42, 15.5), color=PHYS_E, lw=1.2); ax.text(42.7, 13.4, r'$f_\phi(\cdot,p)$', fontsize=6.6, color=PHYS_E, ha='left', va='center')
# p from the split: down the right edge, then left into the physiology box, with a branch up to the readout
line([97, 97], [49.5, 7.25], PHYS_E); arrow((97, 7.25), (57, 7.25), color=PHYS_E, lw=1.1)
arrow((68.5, 7.25), (68.5, 15.5), color=PHYS_E, lw=1.1)
ax.text(97.6, 30, r'$p$', fontsize=7.2, color=PHYS_E, ha='left', va='center')
ax.text(69.2, 11.2, r'$p$', fontsize=6.6, color=PHYS_E, ha='left', va='center')
ax.text(77.0, 4.0, 'same patient, any regimen', ha='center', va='center', fontsize=6.2, color=MUTED, style='italic')

# prediction inset: a regimen with irregular doses
ia = fig.add_axes([0.805, 0.25, 0.135, 0.165])
t = np.linspace(0, 30, 600); doses = [(0, 1.0), (8, 1.0), (13, 0.5), (24, 1.5)]
ka, ke = 1.2, 0.18
c = np.zeros_like(t)
for tk, a in doses:
    s_t = np.clip(t - tk, 0, None)
    c += np.where(t >= tk, a * ka / (ka - ke) * (np.exp(-ke * s_t) - np.exp(-ka * s_t)), 0)
ia.plot(t, c, color=OUT_E, lw=1.1)
for tk, a in doses:
    ia.annotate('', xy=(tk, -0.02), xytext=(tk, -0.02 - 0.18 * a), arrowprops=dict(arrowstyle='-|>', color=DRUG_E, lw=0.9),
                annotation_clip=False)
ia.set_xticks([]); ia.set_yticks([]); ia.set_ylim(-0.05, c.max() * 1.1); ia.set_xlim(-1, 30)
for s_ in ('top', 'right'):
    ia.spines[s_].set_visible(False)
ia.patch.set_alpha(0)
ia.set_xlabel('time (arrows: doses)', fontsize=5.8, labelpad=6)
ax.text(87.5, 29.8, 'exposure under\nany regimen', ha='center', va='bottom', fontsize=6.4, color=OUT_E, fontweight='bold')

fig.savefig(OUT + '.pdf'); fig.savefig(OUT + '.png', dpi=300)
print('wrote', OUT + '.{pdf,png}')
