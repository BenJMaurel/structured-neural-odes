#!/usr/bin/env python3
"""Patient-level view of Baaz's regimen test: for a few TEST patients, the concentration curves under
the training regimen and the 5 unseen ones -- noise-free truth, B version 2 and B as is (posterior-mean latent from the
patient's factual data), the noisy samples the encoder saw (factual panel only), dose times. Patients are picked by
B version 2's mean |AUC err| over the unseen regimens at fixed quantiles (no cherry-picking), shown with their errors.

    plot_regimen_patients.py <n_per_arm> <dense|sparse> <seed> <out.png> [--q 0.25,0.5,0.75,0.95]
Data, patients and models are regenerated / reloaded exactly as in dh_regimen.py / dh_regimen_v2.py (runpy, no training).
Colours: reference palette (dataviz skill), validated: B2 slot 1 blue, B as is slot 2 orange; truth = neutral ink."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, runpy, argparse
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser()
ap.add_argument('n_per_arm'); ap.add_argument('design'); ap.add_argument('seed'); ap.add_argument('out')
ap.add_argument('--q', default='0.25,0.5,0.75,0.95')
cli = ap.parse_args()
os.chdir(REPO); sys.path.insert(0, os.getcwd())
tag = f'N{cli.n_per_arm}_{cli.design}_s{cli.seed}'
SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
C_B2, C_B = '#2a78d6', '#eb6834'
REG_TITLE = {'train': 'Training regimen\n400 or 800 mg at 0/8/16/24 h', 'amt350': 'Unseen amount\n350 mg at 0/8/16/24 h',
             'amt600': 'Unseen amount\n600 mg at 0/8/16/24 h', 'amt850': 'Unseen amount\n850 mg at 0/8/16/24 h',
             'time400': 'Unseen times\n400 mg at 0/3/8 h', 'time800': 'Unseen times\n800 mg at 0/3/8 h'}

# ---- B version 2 and B as is, rebuilt on the same data, weights loaded (no training)
argv = ['x', cli.n_per_arm, cli.design, cli.seed, '/tmp/unused.json', '--no-check']
sys.argv = argv; g2 = runpy.run_path('scripts/regimen/dh_regimen_v2.py', run_name='plot')
ck2 = torch.load(f'results/regimen_dh/B2/{tag}.pt', map_location='cpu', weights_only=False)
g2['encoder'].load_state_dict(ck2['encoder']); g2['osplit'].load_state_dict(ck2['osplit']); g2['lin'].load_state_dict(ck2['lin'])
sys.argv = argv; g1 = runpy.run_path('scripts/regimen/dh_regimen.py', run_name='plot')
ck1 = torch.load(f'results/regimen_dh/B/{tag}.pt', map_location='cpu', weights_only=False)
g1['model'].load_state_dict(ck1['state_dict'])
assert abs(ck2['YMAX'] - g2['YMAX']) < 1e-6 and abs(ck1['YMAX'] - g1['YMAX']) < 1e-6, 'data regenerated differently'
assert np.allclose(g1['y_te'], g2['y_te']), 'the two scripts regenerated different test patients'
EG, YMAX, regs, truth, t_eval = g2['EG'], g2['YMAX'], g2['regs'], g2['truth'], g2['t_eval']
names = list(regs)
pred2, pred1 = {}, {}
with torch.no_grad():
    g2['encoder'].eval(); g1['model'].eval()
    mu2, _ = g2['encoder'](g2['X_te'], g2['tp_samp'], static=g2['S_te'], run_backwards=True)
    mu1, _ = g1['model'].encoder_z0(g1['X_te'], g1['tp_samp'], static=g1['S_te'], run_backwards=True)
    for n in names:
        times, A = regs[n]
        pred2[n] = g2['predict'](mu2, times, A, EG)[0].numpy() * YMAX
        pred1[n] = g1['decode'](g1['simulate'](mu1, times, A, EG))[0].numpy() * YMAX
tru = {n: np.stack([np.interp(EG, t_eval, c) for c in truth[n]]) for n in names}
err = lambda P, n: np.trapezoid(P[n], EG, axis=1) / np.trapezoid(tru[n], EG, axis=1) - 1
e2 = {n: err(pred2, n) for n in names}; e1 = {n: err(pred1, n) for n in names}
# consistency with the run's own JSON (same weights, same patients -> same per-patient errors)
j2 = json.load(open(f'results/regimen_dh/B2/{tag}.json'))
assert np.allclose(e2['amt600'], j2['individual']['amt600']['auc_err_per_patient'], atol=1e-4), 'B2 errors differ from the run'

score = np.mean([np.abs(e2[n]) for n in names[1:]], 0)          # B2 mean |AUC err| over the 5 unseen regimens
qs = [float(x) for x in cli.q.split(',')]
order = np.argsort(score)
picks = [int(order[min(len(order) - 1, int(round(q * (len(order) - 1))))]) for q in qs]

# ---- figure
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF})
R, Cn = len(picks), len(names)
fig, axs = plt.subplots(R, Cn, figsize=(2.55 * Cn, 2.25 * R + 1.2), sharex=True, sharey='row', squeeze=False)
SAMP, y_te, arm = g2['SAMP'], g2['y_te'], g2['arm_te']
for r, i in enumerate(picks):
    for c, n in enumerate(names):
        ax = axs[r, c]
        ax.grid(axis='y', color=GRID, linewidth=0.6); ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.plot(t_eval, truth[n][i], color=INK2, lw=2.2, solid_capstyle='round', zorder=2)
        ax.plot(EG, pred1[n][i], color=C_B, lw=1.6, ls=(0, (4, 2)), zorder=3)
        ax.plot(EG, pred2[n][i], color=C_B2, lw=1.8, zorder=4)
        if n == 'train':
            ax.plot(SAMP, y_te[i], 'o', ms=4.2, mfc=SURF, mec=INK, mew=1.0, zorder=5)
        times, A = regs[n]
        ax.plot(times, np.zeros(len(times)), marker='^', ls='none', ms=6, color=MUTED, clip_on=False, zorder=6)
        ax.text(0.98, 0.95, f'B v2 {100 * e2[n][i]:+.0f}%\nB as is {100 * e1[n][i]:+.0f}%', transform=ax.transAxes,
                ha='right', va='top', fontsize=7.5, color=INK2, linespacing=1.25, zorder=7,
                bbox=dict(facecolor=SURF, edgecolor='none', alpha=0.85, pad=1.5))
        ax.set_xlim(0, 30); ax.set_xticks([0, 8, 16, 24, 30])
        if r == 0:
            ax.set_title(REG_TITLE[n], fontsize=8.5, pad=6)
        if r == R - 1:
            ax.set_xlabel('time (h)')
    pct = int(round(100 * qs[r]))
    axs[r, 0].set_ylabel(f'patient {i + 1}, {int(arm[i])} mg arm\n{pct}th pct of B v2 error ({100 * score[i]:.1f}%)\n'
                         f'concentration (mg/L)', fontsize=8)
handles = [Line2D([], [], color=INK2, lw=2.2, label='truth (noise-free, their simulator)'),
           Line2D([], [], color=C_B2, lw=1.8, label='B version 2 (linear compartments, fixed noise)'),
           Line2D([], [], color=C_B, lw=1.6, ls=(0, (4, 2)), label='B as is'),
           Line2D([], [], marker='o', ls='none', mfc=SURF, mec=INK, mew=1.0, ms=5, label='observed samples (encoder input, training regimen only)'),
           Line2D([], [], marker='^', ls='none', color=MUTED, ms=6, label='dose')]
fig.legend(handles=handles, loc='upper center', ncol=5, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
cfg = {'dense': 'dense sampling (61 per patient)', 'sparse': 'sparse sampling (6 per patient)'}[cli.design]
fig.suptitle(f"Baaz et al.'s regimen test, {cli.n_per_arm} patients per training arm, {cfg}, seed {cli.seed}: "
             f"4 test patients at the {', '.join(str(int(round(100 * q))) for q in qs)}th percentiles of B version 2's error",
             y=0.955, fontsize=10, color=INK)
fig.text(0.5, 0.005, 'Each patient is encoded once from its training-regimen samples (left column); every other column is a '
         'prediction for a regimen the model never saw, with no new samples. Panel text: AUC(0-30 h) error vs truth.',
         ha='center', fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 0.93))
os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
fig.savefig(cli.out, dpi=160, facecolor=SURF)
print('saved', cli.out, '| patients', [p + 1 for p in picks], '| B2 mean |err| unseen',
      [f'{100 * score[p]:.1f}%' for p in picks])
