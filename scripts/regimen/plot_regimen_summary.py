#!/usr/bin/env python3
"""Summary of Baaz's regimen test: median individual |AUC(0-30 h) error| per regimen, EB-VAE vs B as is
vs B version 2, small multiples over the 6 configurations (rows: dense / sparse sampling; columns: 25 / 100 / 250
patients per training arm). Dense: seed 0 (the only seed run). Sparse: mean of seeds 0-2, bar = range over seeds.
The plain latent ODE (no B structure) is left out for readability (worse everywhere but one configuration; see
results/regimen_dh/regimen_table.txt). Colours: reference palette slots 1-3, validated all-pairs (aqua < 3:1 contrast
-> relief = the table view results/regimen_dh/regimen_table.txt + distinct marker shapes), same entity colours as plot_regimen_patients.py.
    plot_regimen_summary.py <out.png>"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
os.chdir(REPO)
out = sys.argv[1] if len(sys.argv) > 1 else 'results/regimen_dh/figures/summary.png'
EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen/res'
SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
MODELS = (('ebvae', 'EB-VAE (Baaz 2026)', '#1baf7a', 'D'), ('B', 'B as is', '#eb6834', 's'), ('B2', 'B version 2', '#2a78d6', 'o'))
REGS = ('train', 'amt350', 'amt600', 'amt850', 'time400', 'time800')
XLAB = ('training\nregimen', '350 mg', '600 mg', '850 mg', '400 mg\n0/3/8 h', '800 mg\n0/3/8 h')


def load(m, n, d, s):
    p = (f'{EB}/N{n}_{d}' + (f'_s{s}' if s else '') + '.json') if m == 'ebvae' else f'results/regimen_dh/{m}/N{n}_{d}_s{s}.json'
    return json.load(open(p)) if os.path.exists(p) else None


plt.rcParams.update({'font.size': 9, 'axes.edgecolor': AXIS, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.labelcolor': INK2, 'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.4), sharey='row')
x = np.arange(len(REGS)); off = {'ebvae': -0.22, 'B': 0.0, 'B2': 0.22}
for r, d in enumerate(('dense', 'sparse')):
    for c, n in enumerate((25, 100, 250)):
        ax = axs[r, c]
        ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.axvspan(0.5, len(REGS) - 0.5, color='#f0efec', zorder=0, lw=0)          # unseen regimens
        seeds = (0,) if d == 'dense' else (0, 1, 2)
        for m, lab, col, mk in MODELS:
            runs = [load(m, n, d, s) for s in seeds]
            runs = [q for q in runs if q is not None]
            if not runs:
                continue
            v = np.array([[q['individual'][g]['auc_abs_median'] for g in REGS] for q in runs])
            mean = v.mean(0)
            if len(runs) > 1:
                ax.vlines(x + off[m], v.min(0), v.max(0), color=col, lw=1.4, alpha=0.6, zorder=2)
            ax.plot(x + off[m], mean, mk, color=col, ms=7, mec=SURF, mew=1.2, zorder=3, ls='none')
        ax.set_xticks(x); ax.set_xticklabels(XLAB, fontsize=7.5)
        ax.set_xlim(-0.6, len(REGS) - 0.4)
        ax.set_title(f'{n} patients per arm, {d}' + (' (61 samples)' if d == 'dense' else ' (6 samples)'), fontsize=9)
        if c == 0:
            ax.set_ylabel('median individual |AUC(0-30 h) error| (%)')
        ax.set_ylim(bottom=0)
handles = [plt.Line2D([], [], marker=mk, ls='none', color=col, ms=7, mec=SURF, label=lab) for _, lab, col, mk in MODELS]
fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Baaz et al.'s regimen test: 400 test patients encoded from their training-regimen data, predicted under "
             "unseen regimens (shaded)", y=0.955, fontsize=10, color=INK)
fig.text(0.5, 0.008, 'Dense: seed 0 only. Sparse: mean of seeds 0-2, vertical bar = range over seeds. Same patients for '
         'all models. Plain latent ODE omitted. Numbers: results/regimen_dh/regimen_table.txt.', ha='center', fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 0.93))
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
fig.savefig(out, dpi=160, facecolor=SURF)
print('saved', out)
