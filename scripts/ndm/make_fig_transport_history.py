#!/usr/bin/env python3
"""Step-3 figure of ndm.tex, revised (). Previous version:
make_fig_transport_history.py.bak_v1 (panel b = dose-event model over seven doses, dropped).
  a  one test patient (4 -> 8 mg): OT-FiLM's predicted concentration at the current dose, at the new dose with the transport applied at
     the dose (what it is trained for), and with the same transport applied 3, 6 or 12 h later to the state reached by then
  b  all Advagraf test patients: AUC over the rest of the interval when the transport is applied at time t, relative to applying it at
     the dose and following the flow (median, 10-90 % band); 1 = what an action on the drug state would give
  c  linear read-out of the log central volume from the latent state: learned at the dose and applied at later times; learned again
     at each time; and, for the dose-event model, from its physiology coordinates (one read-out for every time)
Data: results/ndm/fig_transport_at_t.npz (fig_transport_at_t_data.py) and results/ndm/fig_transport_history.npz (panel c;
fig_transport_history_data.py). Writes figures/transport_vs_events.{pdf,png} in the manuscript folder."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = np.load('results/ndm/fig_transport_at_t.npz'); H = np.load('results/ndm/fig_transport_history.npz')
OUT = f'{REPO}/figures/transport_vs_events'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'mathtext.fontset': 'dejavusans', 'axes.linewidth': 0.7,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5})
TEAL, PURPLE, INK, MUTED, BLUE = '#12807b', '#6a3d9a', '#222222', '#666666', '#1f5673'
LATER = {3.0: '#e08a1e', 6.0: '#c8313a'}   # 22/10: 12 h dropped from panel a (gamma > 0 seed 1 reaches 150 ng/mL there; panel b covers it)

fig = plt.figure(figsize=(6.4, 2.35))
gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.0], wspace=0.42, left=0.065, right=0.985, top=0.86, bottom=0.19)

# ---------------------------------------------------------------- a: one patient
ax = fig.add_subplot(gs[0])
tg = A['tg']
ax.plot(tg, A['ex_old'], color=INK, lw=1.1, label=f'current dose ({A["ex_d1"]:.0f} mg)')
ax.plot(tg, A['ex_new0'], color=BLUE, lw=1.2, ls=(0, (3, 1.5)), label=f'alternative dose ({A["ex_d2"]:.0f} mg),\ntransport at the dose')
for tau, col in LATER.items():
    k = int(np.argmin(np.abs(tg - tau)))
    cur = A[f'ex_tau{tau:g}']
    ax.plot(tg[k:], cur, color=col, lw=1.1, label=f'same transport at {tau:g} h')
    ax.plot([tg[k]], [cur[0]], 'o', color=col, ms=3)
ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24]); ax.set_ylim(0, 1.36 * max([A['ex_new0'].max(), A['ex_old'].max()] + [A[f'ex_tau{t:g}'].max() for t in LATER]))
ax.set_xlabel('hours after the dose'); ax.set_ylabel('predicted concentration (ng/mL)', labelpad=1)
ax.legend(fontsize=5.4, frameon=False, loc='upper right', handlelength=1.6, labelspacing=0.3, bbox_to_anchor=(1.02, 1.04))
ax.set_title('a   one patient', fontsize=7.5, loc='left', fontweight='bold')

# ---------------------------------------------------------------- b: all patients, every seed (22/10: the Table 2 models have gamma > 0
# and their transferability varies across seeds, so panel b shows each of the ten seeds and their median, not seed 1 alone)
ax = fig.add_subplot(gs[1])
ta = A['taus']
SEEDS = sorted(glob.glob(os.environ.get('FIG_SEEDS_GLOB', 'results/ndm/fig_transport_at_t_seeds/s4_pnoise_win_gpos_s*.npz')))
M = np.array([np.load(p)['med_abs_pct'] for p in SEEDS])
assert len(SEEDS) == 10 and np.allclose(M[[k for k, p in enumerate(SEEDS) if p.endswith('_s1.npz')][0]], A['med_abs_pct'])
for k, m in enumerate(M):
    ax.plot(ta, m, color=TEAL, lw=0.6, alpha=0.35, label='each training seed' if k == 0 else None)
ax.plot(ta, np.median(M, 0), color=TEAL, lw=1.5, marker='o', ms=2.5, label='median over seeds')
ax.axhline(0.0, color=INK, lw=0.7, ls=':')
ax.text(23.5, 2.5, 'same as at the dose', fontsize=5.8, color=INK, ha='right', va='bottom')
ax.set_ylim(-3, 105); ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24])
ax.set_xlabel('time t at which the transport is applied (h)')
ax.set_ylabel('AUC from t to 24 h,\nmedian |difference| (%)', labelpad=1)
ax.legend(fontsize=5.6, frameon=False, loc='upper left', handlelength=1.6)
ax.set_title('b   all test patients', fontsize=7.5, loc='left', fontweight='bold')
print('panel b, median over seeds of the median |r-1| %:', dict(zip(ta.tolist(), np.round(np.median(M, 0), 1).tolist())),
      '| range at 3 / 6 / 12 h:', [f'{M[:, list(ta).index(t)].min():.0f}-{M[:, list(ta).index(t)].max():.0f}' for t in (3.0, 6.0, 12.0)])

# ---------------------------------------------------------------- c: read-out of the patient
ax = fig.add_subplot(gs[2])
t = H['tgrid']; fixed, refit = H['film_fixed'], H['film_refit']
ax.plot(t, np.clip(fixed, -1.0, None), color=TEAL, lw=1.2, label='OT-FiLM, read-out learned\nat the dose, applied later')
ax.plot(t, refit, color=TEAL, lw=1.0, ls=(0, (3, 1.5)), label='OT-FiLM, a new read-out\nlearned at each time')
ax.plot(t, np.full_like(t, float(H['event_r2p'])), color=PURPLE, lw=1.2, label='dose-event model, one\nread-out of $p$ for all times')
ax.axhline(0, color=MUTED, lw=0.5, ls=':')
ax.annotate(f'{fixed[2]:.0f} at 1 h', xy=(t[2], -1.0), xytext=(5.5, -0.78), fontsize=5.8, color=TEAL,
            arrowprops=dict(arrowstyle='-|>', lw=0.6, color=TEAL))
ax.set_ylim(-1.05, 0.45); ax.set_xlim(0, 24); ax.set_xticks([0, 6, 12, 18, 24])
ax.set_xlabel('hours after the dose'); ax.set_ylabel('$R^2$, log central volume', labelpad=1)
ax.legend(fontsize=5.5, frameon=False, loc='center right', bbox_to_anchor=(1.03, 0.43), handlelength=1.8, labelspacing=0.6)
ax.set_title('c   reading the patient', fontsize=7.5, loc='left', fontweight='bold')
for a_ in fig.axes:
    for s_ in ('top', 'right'):
        a_.spines[s_].set_visible(False)

for ext in ('pdf', 'png'):
    fig.savefig(f'{OUT}.{ext}', dpi=300, facecolor='white')
print(f'wrote {OUT}.{{pdf,png}}')
