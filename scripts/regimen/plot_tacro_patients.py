#!/usr/bin/env python3
"""Patient-level view of the tacrolimus regimen benchmark: what the dose-event model receives and what it returns.
For a few TEST patients (picked by rule: quantiles of the model's mean |AUC error| over the five unseen regimens, no choice by eye),
one row each: the noisy samples the encoder saw under the patient's training regimen, then the predicted curve (posterior-mean
latent) against the noise-free truth under the training regimen and the five unseen ones, with the AUC(0-45 h) error.
Data, patients and model are regenerated / reloaded exactly as in dh_regimen.py (runpy with the run's own flags, no training);
the reload is checked by recomputing the run's median errors from the JSON.
    plot_tacro_patients.py [design sparse|dense] [seed] [--q 0.25,0.5,0.75,0.95] [--root results/regimen_dh/tacro/Bfixkp4n]
    -> <root>/../figures/patients_<design>_s<seed>.png"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, runpy, argparse
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ap = argparse.ArgumentParser()
ap.add_argument('design', nargs='?', default='sparse'); ap.add_argument('seed', nargs='?', type=int, default=0)
ap.add_argument('--q', default='0.25,0.5,0.75,0.95')
ap.add_argument('--root', default='results/regimen_dh/tacro/Bfixkp4n')
cli = ap.parse_args()
os.chdir(REPO); sys.path.insert(0, os.getcwd())
tag = f'N100_{cli.design}_s{cli.seed}'
ck = torch.load(f'{cli.root}/{tag}.pt', map_location='cpu', weights_only=False)
c = ck['cli']
argv = ['x', '100', cli.design, str(cli.seed), '/tmp/unused.json', '--sim', c['sim'], '--enc-dose', c['enc_dose']]
argv += (['--obsrv-mgl', str(c['obsrv_mgl'])] if c.get('obsrv_mgl') is not None else [])
argv += (['--kick-phys', str(c['kick_phys'])] if c.get('kick_phys') else [])
argv += (['--titrate', str(c['titrate'])] if c.get('titrate') is not None else [])
sys.argv = argv; g = runpy.run_path('scripts/regimen/dh_regimen.py', run_name='plot')
g['model'].load_state_dict(ck['state_dict']); g['model'].eval()
assert abs(ck['YMAX'] - g['YMAX']) < 1e-9, 'data regenerated differently'
G, EG, YMAX, regs, truth, t_eval = g['G'], g['EG'], g['YMAX'], g['regs'], g['truth'], g['t_eval']
names = list(regs)

# ---- predictions for every test patient (as dh_regimen.main), checked against the run's JSON
with torch.no_grad():
    mu, _ = g['model'].encoder_z0(g['X_te'], g['tp_samp'], static=g['S_te'], run_backwards=True)
    pred = {n: g['decode'](g['simulate'](mu, regs[n][0], regs[n][1], EG))[0].numpy() * YMAX for n in names}
tru = {n: np.stack([np.interp(EG, t_eval, cc) for cc in truth[n]]) for n in names}
err = {n: np.trapezoid(pred[n], EG, axis=1) / np.trapezoid(tru[n], EG, axis=1) - 1 for n in names}
ref = json.load(open(f'{cli.root}/{tag}.json'))['individual']
for n in names:
    assert abs(100 * np.median(np.abs(err[n])) - ref[n]['auc_abs_median']) < 1e-3, f'{n}: reload does not reproduce the run'
print('reload reproduces the run JSON (median |AUC err| per regimen)')

unseen = [n for n in names if n != 'train']
score = np.mean(np.abs(np.stack([err[n] for n in unseen])), 0)
qs = [float(q) for q in cli.q.split(',')]
order = np.argsort(score); picks = [int(order[min(int(round(q * (len(score) - 1))), len(score) - 1)]) for q in qs]

# ---- figure
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3de'
C_MOD, C_OBS = '#2a78d6', '#eb6834'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 8.8, 'axes.titleweight': 'bold',
                     'axes.titlecolor': INK, 'axes.titlelocation': 'left'})
LO, HI = G.ARMS
sched = lambda t: '/'.join(f'{x:g}' for x in t)
TITLE = {'train': f'Observed regimen (input)\n{LO:g} or {HI:g} mg at {sched(G.TRAIN_TIMES)} h'}
for n in unseen:
    times, A = regs[n]
    TITLE[n] = f'Unseen: {A[0, 0]:g} mg\nat {sched(times)} h'
fig, axes = plt.subplots(len(picks), len(names), figsize=(2.25 * len(names), 2.05 * len(picks) + 0.9), sharex=True)
SAMP = g['SAMP']
for r, i in enumerate(picks):
    ymax = 1.08 * max(max(tru[n][i].max(), pred[n][i].max()) for n in names)
    ymax = max(ymax, 1.08 * g['y_te'][i].max())
    for k, n in enumerate(names):
        ax = axes[r, k]; times, A = regs[n]
        ax.plot(EG, tru[n][i], color=INK, lw=1.4, zorder=2)
        ax.plot(EG, pred[n][i], color=C_MOD, lw=1.6, ls=(0, (4, 1.6)), zorder=3)
        if n == 'train':
            ax.plot(SAMP, g['y_te'][i], 'o', color=C_OBS, ms=4.2, zorder=4, mec='white', mew=0.6)
        for t, a in zip(times, A[i]):
            ax.plot([t], [0], marker='^', color=INK2, ms=4.5, clip_on=False, zorder=5)
        ax.set_ylim(0, ymax); ax.set_xlim(0, G.T_END); ax.set_xticks([0, 12, 24, 36, 45])
        ax.grid(axis='y', color=GRID, lw=0.6)
        ax.text(0.97, 0.95, f'AUC {100 * err[n][i]:+.0f}%', transform=ax.transAxes, ha='right', va='top', fontsize=8,
                color=C_MOD if abs(err[n][i]) <= 0.2 else '#c8313a', fontweight='bold')
        if r == 0:
            ax.set_title(TITLE[n], fontsize=8.3)
        if k == 0:
            P = g['P_te'][i]
            ax.set_ylabel(f'{100 * qs[r]:g}th pct patient\n{g["arm_te"][i]:g} mg arm, CYP3A5 {"exp." if P[6] else "non-exp."}\n'
                          f'Ht {P[7]:.0f}%  |  ng/mL', fontsize=7.8)
        if r == len(picks) - 1:
            ax.set_xlabel('hours')
handles = [Line2D([], [], color=INK, lw=1.4, label='true concentration (noise-free)'),
           Line2D([], [], color=C_MOD, lw=1.6, ls=(0, (4, 1.6)), label='dose-event model prediction'),
           Line2D([], [], marker='o', color=C_OBS, ls='', ms=4.5, label=f'samples given to the encoder ({len(SAMP)}, noisy)'),
           Line2D([], [], marker='^', color=INK2, ls='', ms=5, label='dose')]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=8.3, bbox_to_anchor=(0.5, 0.0))
fig.suptitle(f'Tacrolimus regimen benchmark, {cli.design} sampling, seed {cli.seed}: test patients at the '
             f'{", ".join(f"{100 * q:g}" for q in qs)}th percentiles of the mean |AUC error| over unseen regimens',
             fontsize=9.5, x=0.01, ha='left', fontweight='bold', color=INK)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))
out = os.path.join(os.path.dirname(cli.root.rstrip('/')), 'figures', f'patients_{cli.design}_s{cli.seed}.png')
os.makedirs(os.path.dirname(out), exist_ok=True); fig.savefig(out, dpi=170)
print('wrote', out)
for r, i in enumerate(picks):
    P = g['P_te'][i]
    print(f'  q{qs[r]:.2f}: test patient {i}, arm {g["arm_te"][i]:g} mg, Vmax {P[4]:.3f} mg/h, Km {1000 * P[5]:.1f} ng/mL, Vc {P[2]:.0f} L, '
          f'mean |err| unseen {100 * score[i]:.1f}%  ' + '  '.join(f'{n} {100 * err[n][i]:+.0f}' for n in names))
