#!/usr/bin/env python3
"""Per-patient AUC errors pooled over the 100 runs of a scenario, latent ODE vs MAP-BE (published init, form fix),
and how they depend on the degree of saturation. Scenario 3 = Michaelis-Menten elimination, Km_pop = 0.01 mg/L =
10 ng/mL (IPV sd 0.32 on log Km; individual Km not saved), so the patient's average concentration over the observed
interval, Cavg = true AUC / tau (ng/mL), relative to 10 ng/mL is the saturation index. Scenario 1 (linear data) is the
control: any trend with Cavg there is not nonlinearity.   error_by_saturation.py [scenarios] -> text + figure"""
import os, sys, glob, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__)); from summarise import load
OUT = 'results/scen_rerun/error_by_saturation'; os.makedirs(OUT, exist_ok=True)
def pooled(sc):
    rows = []
    for d in sorted(glob.glob(f'results/scen_rerun/runs/s{sc}_seed[0-9][0-9][0-9]')):
        if not (os.path.exists(f'{d}/DONE') and os.path.exists(f'{d}/DONE_c1')): continue
        b, _ = load(d); x = pd.read_csv(f'{d}/tacro_mapbayest_auc_sigc1.csv').set_index('ID')
        t = pd.read_csv(f'{d}/virtual_cohort_test.csv').groupby('ID').first()
        ids = sorted(set(b['lode_best'].index) & set(x.index))
        for i in ids:
            tau = float(t.loc[i, 'II']); auc = float(x.loc[i, 'AUC_observed'])
            rows.append(dict(run=os.path.basename(d), ID=i, drug=t.loc[i, 'DRUG'], tau=tau, auc=auc, cavg=auc / tau,
                             lode_best=b['lode_best'][i], lode_final=b['lode_final'][i], mapbe=x.loc[i, 'auc_ipred'] / auc - 1))
    return pd.DataFrame(rows)
scens = [int(s) for s in (sys.argv[1].split() if len(sys.argv) > 1 else ['3', '1'])]
EDGES = [0, 2.5, 5, 10, 20, 40, np.inf]          # Cavg ng/mL; Km_pop = 10
M = (('lode_best', 'Latent ODE (_best)'), ('lode_final', 'Latent ODE (final)'), ('mapbe', 'MAP-BE (published init)'))
res = {}
for sc in scens:
    D = pooled(sc); D.to_csv(f'{OUT}/s{sc}_per_patient.csv', index=False); res[sc] = D
    print('=' * 104); print(f'SCENARIO {sc}: {len(D)} patients from {D.run.nunique()} runs; Cavg median {D.cavg.median():.1f} ng/mL '
          f'[5-95%: {D.cavg.quantile(.05):.1f}-{D.cavg.quantile(.95):.1f}]; share with Cavg > Km_pop (10): {100 * (D.cavg > 10).mean():.0f}%')
    print('  pooled |error| quantiles (%)          p50    p75    p90    p95    p99    max | MPE  | >20%  >50%')
    for k, lab in M:
        a = 100 * D[k].abs()
        print(f'  {lab:32s} ' + ' '.join(f'{a.quantile(q):6.1f}' for q in (.5, .75, .9, .95, .99)) + f' {a.max():6.1f} | {100 * D[k].mean():+4.1f} | '
              f'{(a > 20).sum():4d} {(a > 50).sum():4d}')
    D['bin'] = pd.cut(D.cavg, EDGES, right=False)
    print('  by saturation (Cavg/Km_pop):  n   | median |err|: LODE_best  MAP-BE | p90 |err|: LODE_best  MAP-BE | MPE: LODE_best  MAP-BE | LODE closer')
    for bn, g in D.groupby('bin', observed=True):
        lo, hi = bn.left, bn.right
        print(f'   Cavg {lo:4.1f}-{hi:5.1f} ({lo / 10:3.2f}-{hi / 10:4.1f}x Km) {len(g):5d} | {100 * g.lode_best.abs().median():8.1f} {100 * g.mapbe.abs().median():7.1f} | '
              f'{100 * g.lode_best.abs().quantile(.9):8.1f} {100 * g.mapbe.abs().quantile(.9):7.1f} | {100 * g.lode_best.mean():+8.1f} {100 * g.mapbe.mean():+7.1f} | '
              f'{100 * (g.lode_best.abs() < g.mapbe.abs()).mean():5.1f}%')
    for drug, g in D.groupby('drug'):
        print(f'   {drug:9s} n={len(g):4d}  median |err| LODE_best {100 * g.lode_best.abs().median():4.1f}  MAP-BE {100 * g.mapbe.abs().median():4.1f}  '
              f'MPE {100 * g.lode_best.mean():+4.1f} / {100 * g.mapbe.mean():+4.1f}')

# ---- figure: per scenario, median and 90th-percentile |error| by saturation bin (one y-axis per panel) ----
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
C = {'lode_best': '#2a78d6', 'mapbe': '#eb6834'}; INK, INK2, SURF = '#0b0b0b', '#52514e', '#fcfcfb'
fig, axes = plt.subplots(1, len(scens), figsize=(5.6 * len(scens), 4.2), facecolor=SURF, sharey=True)
axes = np.atleast_1d(axes)
for ax, sc in zip(axes, scens):
    D = res[sc]; D['bin'] = pd.cut(D.cavg, EDGES, right=False)
    g = [(bn, v) for bn, v in D.groupby('bin', observed=True) if len(v) >= 30]; x = np.arange(len(g)); labs = []   # bins with < 30 patients not plotted
    for k, lab, dx, mk in (('lode_best', 'Latent ODE', -0.08, 'o'), ('mapbe', 'MAP-BE (published init)', 0.08, 's')):
        med = [100 * v[k].abs().median() for _, v in g]; p90 = [100 * v[k].abs().quantile(.9) for _, v in g]
        ax.plot(x + dx, med, '-', marker=mk, ms=8, lw=2, color=C[k], label=f'{lab}: median', mec=SURF, mew=2)
        ax.plot(x + dx, p90, '--', marker=mk, ms=7, lw=2, color=C[k], alpha=.55, label=f'{lab}: 90th percentile', mec=SURF, mew=2)
    for bn, v in g: labs.append(f'{bn.left:g}-{bn.right:g}\n(n={len(v)})' if np.isfinite(bn.right) else f'≥{bn.left:g}\n(n={len(v)})')
    xk = EDGES.index(10) - 0.5
    if sc == 3: ax.axvline(xk, color=INK2, lw=1, ls=':'); ax.text(xk + 0.05, 1, 'Km (population) = 10 ng/mL', color=INK2, fontsize=8, va='bottom')
    ax.set_xticks(x, labs, fontsize=8, color=INK2); ax.set_xlabel('Average concentration over the interval, ng/mL', color=INK2, fontsize=9)
    ax.set_title(f'Scenario {sc}' + (' — saturable elimination' if sc == 3 else ' — linear (control)'), color=INK, fontsize=11, loc='left')
    ax.set_facecolor(SURF); ax.grid(axis='y', color='#e4e3dd', lw=.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color('#c3c2b7')
    ax.tick_params(colors=INK2, labelsize=8)
axes[0].set_ylabel('|AUC error|, %', color=INK2, fontsize=9); axes[0].set_ylim(bottom=0)
axes[-1].legend(frameon=False, fontsize=8, labelcolor=INK2, loc='upper left')
fig.tight_layout(); fig.savefig(f'{OUT}/error_by_saturation.png', dpi=160, facecolor=SURF); print(f'figure -> {OUT}/error_by_saturation.png')
