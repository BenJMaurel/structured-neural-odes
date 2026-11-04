#!/usr/bin/env python3
"""Main-text patient figure of ndm.tex (). One test patient of the tacrolimus
regimen benchmark (linear elimination, six samples, seed 0; rule and checks in fig_tacro_patients_data.py). Rows: models. Columns: the
training regimen, whose six samples are all the models see, then an unseen dose and an unseen schedule predicted from those samples.
Data: results/ndm/fig_tacro_patients[_visit].npz. Writes figures/tacro_patients[_visit].{pdf,png} in the manuscript folder.
    make_fig_tacro_patients.py [sparse|visit]   (19/10: visit = intensive PK visit, 12 samples)"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import sys
DESIGN = sys.argv[1] if len(sys.argv) > 1 else 'sparse'
SFX = ('' if DESIGN == 'sparse' else f'_{DESIGN}') + ('_depot' if 'depot' in sys.argv[2:] else '')   # 19/10: make_fig_tacro_patients.py visit depot
SFX += next((f'_{a}' for a in sys.argv[2:] if a.startswith('B')), '')   # 22/10: run directory of the dose-event model (e.g. Bv1sp20)
SFX += '_clip0' if 'clip0' in sys.argv[2:] else ''
SFX += '_luisfact' if 'luisfact' in sys.argv[2:] else ''   # 22/10: Lu regimen model with the factual encoder   # 20/10: make_fig_tacro_patients.py visit clip0 (every model clipped at zero)
D = np.load(f'results/ndm/fig_tacro_patients{SFX}.npz')
OUT = f'{REPO}/figures/tacro_patients{SFX}'
NSAMP = {6: 'six', 12: 'twelve'}.get(len(D['samp']), str(len(D['samp'])))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'mathtext.fontset': 'dejavusans', 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'xtick.major.size': 2.2, 'ytick.major.size': 2.2})
INK, MUTED, SAMPLE = '#222222', '#8a8a8a', '#d9822b'
MODELS = [('dose-event', 'Dose-event model\n(ours)', '#6a3d9a'), ('EB-VAE', 'EB-VAE', '#1b9e77'),
          ('Lu iScience', 'Lu et al.,\nregimen model', '#1f5673'), ('Lu NMI', 'Lu et al.,\ntwo-port model', '#8c564b')]
fmt = lambda a: f'{a:g}'
arm = float(D['arm'])
COLS = [('train', f'a   Training regimen, observed\n{fmt(arm)} mg at 0, 12, 24, 36 h'),
        ('amt4.25', 'b   Unseen dose\n4.25 mg at 0, 12, 24, 36 h'),
        ('timeLOW', f'c   Unseen schedule\n{fmt(arm)} mg at 0, 4.5, 12 h')]
EG = D['EG']
ymax = 1.28 * max(max(D[f'truth_{r}'].max(), *(D[f'pred_{k}_{r}'].max() for k, _, _ in MODELS)) for r, _ in COLS)
# 19/10: never clip negative predictions (the 12-sample design leaves the first 12 h unobserved; some models start below zero)
ymin = min(0.0, 1.15 * min(min(D[f'pred_{k}_{r}'].min() for k, _, _ in MODELS) for r, _ in COLS))

fig, axes = plt.subplots(len(MODELS), len(COLS), figsize=(6.4, 4.55), sharex=True, sharey=True)
fig.subplots_adjust(left=0.115, right=0.99, top=0.85, bottom=0.085, wspace=0.08, hspace=0.18)
for j, (r, title) in enumerate(COLS):
    times, amts = D[f'times_{r}'], D[f'amounts_{r}']
    for i, (k, lab, col) in enumerate(MODELS):
        ax = axes[i, j]
        ax.plot(EG, D[f'truth_{r}'], color=INK, lw=1.0, zorder=3)
        ax.plot(EG, D[f'pred_{k}_{r}'], color=col, lw=1.3, ls=(0, (3.2, 1.6)), zorder=4)
        ax.plot(times, np.full(len(times), ymin + 0.035 * (ymax - ymin)), ls='none', marker='^', ms=3.2, color=MUTED, zorder=2, clip_on=False)
        if r == 'train':
            ax.plot(D['samp'], D['y_obs'], ls='none', marker='o', ms=3.0, mfc='white', mec=SAMPLE, mew=1.0, zorder=5)
        e = 100 * float(D[f'err_{k}_{r}'])
        ax.text(0.98, 0.93, f'AUC error {e:+.0f}%', transform=ax.transAxes, ha='right', va='top', fontsize=6.3, color=col)
        ax.set_xlim(0, 45); ax.set_ylim(ymin, ymax)
        if ymin < 0:
            ax.axhline(0, color=MUTED, lw=0.5, zorder=1); ax.set_xticks([0, 12, 24, 36, 45])
        ax.grid(axis='y', color='#e6e6e6', lw=0.5); ax.set_axisbelow(True)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
        if j == 0:
            ax.set_ylabel(lab, fontsize=6.8, labelpad=3)
        if i == len(MODELS) - 1:
            ax.set_xlabel('hours')
    axes[0, j].set_title(title, fontsize=7, loc='left', fontweight='bold', pad=4)
fig.text(0.012, 0.47, 'tacrolimus concentration (ng/mL)', rotation=90, va='center', ha='left', fontsize=7)
handles = [Line2D([], [], color=INK, lw=1.0, label='truth (noise-free)'),
           Line2D([], [], color=MUTED, lw=1.3, ls=(0, (3.2, 1.6)), label='prediction of the model in the row'),
           Line2D([], [], ls='none', marker='o', ms=3.0, mfc='white', mec=SAMPLE, mew=1.0, label=f'the {NSAMP} samples given to every model'),
           Line2D([], [], ls='none', marker='^', ms=3.2, color=MUTED, label='doses')]
fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, fontsize=6.2, bbox_to_anchor=(0.55, 0.997), handlelength=1.8,
           columnspacing=1.0, handletextpad=0.5)
for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}.{ext}', dpi=300, facecolor='white')
print(f'wrote {OUT}.{{pdf,png}}; patient {int(D["patient"])}, arm {arm:g} mg, seed {int(D["seed"])}')
