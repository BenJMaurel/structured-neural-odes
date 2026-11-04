#!/usr/bin/env python3
"""Latent ODE vs MAP-BE with Monolix DEFAULT initial values + combined1 (form-fixed) residual error, per scenario
(27-28/09 runs; chain_scen_c1_definit.sh). Per run (40 test patients; the patients MAP-BE returned):
RMSPE, MPE, median |error| (%). Summaries over runs: median [IQR] and mean +- s.d.; paired LODE - MAP-BE differences
with Wilcoxon signed-rank p and the number of runs where the latent ODE is lower. A run whose default-init MAP-BE
aborted counts as a MAP-BE failure (reported, excluded from the pairs). Fits split into plausible (Vc_pop 100-2000 L,
KTR_pop 1-10 /h) and degenerate. Reference row: MAP-BE published init + form fix on the same patients."""
import os, glob, numpy as np, pandas as pd
from scipy import stats
import sys; sys.path.insert(0, os.path.dirname(__file__))
from summarise import load
R = 'results/scen_rerun/runs'
f3 = lambda x: (100 * np.sqrt(np.mean(x ** 2)), 100 * np.mean(x), 100 * np.median(np.abs(x)))
def rd(d, tag):
    x = pd.read_csv(f'{d}/tacro_mapbayest_auc_{tag}.csv'); return pd.Series((x.auc_ipred / x.AUC_observed - 1).values, index=x.ID.values)
def q(v): return f'{np.median(v):5.1f} [{np.percentile(v, 25):4.1f}-{np.percentile(v, 75):4.1f}]  {np.mean(v):5.1f} +- {np.std(v, ddof=1):4.1f}'
for sc in (1, 2, 3):
    dirs = sorted(d for d in glob.glob(f'{R}/s{sc}_seed[0-9][0-9][0-9]') if os.path.exists(f'{d}/DONE'))
    ok = [d for d in dirs if os.path.exists(f'{d}/DONE_definit_c1') and os.path.exists(f'{d}/DONE_c1')]
    rows = {k: [] for k in ('lode_best', 'lode_final', 'pub_c1', 'def_c1')}; plaus = []
    for d in ok:
        b, _ = load(d); dc = rd(d, 'definit_sigc1'); pc = rd(d, 'sigc1')
        ids = sorted(set(dc.index) & set(b['lode_best'].index) & set(pc.index))
        for k, s in (('lode_best', b['lode_best']), ('lode_final', b['lode_final']), ('pub_c1', pc), ('def_c1', dc)):
            rows[k].append(f3(s.loc[ids].values))
        pf = f'{d}/populationParameters_definit.txt'
        if not os.path.exists(pf):   # default-init var/sd MAP-BE crashed: the fit is only in the slot worktree
            sc_, sd_ = os.path.basename(d)[1], int(os.path.basename(d)[-3:])
            pf = glob.glob(f'results/paper_repro/w*/results/{int(sc_) * 10000 + sd_}_definit/2_test_tacro/populationParameters.txt')[0]
        pp = pd.read_csv(pf).set_index('parameter').value
        plaus.append(100 <= pp['Vc_pop'] <= 2000 and 1 <= pp['KTR_pop'] <= 10)
    A = {k: np.array(v) for k, v in rows.items()}; plaus = np.array(plaus)
    print('=' * 104); print(f'SCENARIO {sc}: {len(ok)}/{len(dirs)} runs with default-init MAP-BE ({len(dirs) - len(ok)} MAP-BE failures); '
          f'plausible default-init fits {plaus.sum()}/{len(plaus)}')
    print(f'  {"":34s} {"RMSPE median [IQR]   mean +- sd":36s} | {"MPE mean +- sd":14s} | median|err| median')
    for k, lab in (('lode_best', 'Latent ODE (_best)'), ('lode_final', 'Latent ODE (final epoch)'),
                   ('def_c1', 'MAP-BE default init + form fix'), ('pub_c1', 'MAP-BE published init + form fix')):
        print(f'  {lab:34s} {q(A[k][:, 0])} | {A[k][:, 1].mean():+5.2f} +- {A[k][:, 1].std(ddof=1):4.2f} | {np.median(A[k][:, 2]):5.1f}')
    for lo in ('lode_best', 'lode_final'):
        for comp in ('def_c1', 'pub_c1'):
            for mi, met in ((0, 'RMSPE'), (2, 'median|err|')):
                dx = A[lo][:, mi] - A[comp][:, mi]; w = stats.wilcoxon(dx)
                print(f'  {lo:10s} - {comp:6s} {met:11s}: median diff {np.median(dx):+6.2f}, mean {dx.mean():+6.2f}; LODE lower in {(dx < 0).sum():3d}/{len(dx)}; Wilcoxon p = {w.pvalue:.2g}')
    for name, m in (('plausible', plaus), ('degenerate', ~plaus)):
        if m.sum() < 2: continue
        dx = A['lode_best'][m, 0] - A['def_c1'][m, 0]
        print(f'  [{name:10s} default-init fits, n={m.sum():3d}] RMSPE MAP-BE median {np.median(A["def_c1"][m, 0]):5.1f} vs LODE _best {np.median(A["lode_best"][m, 0]):5.1f}; '
              f'LODE lower in {(dx < 0).sum()}/{m.sum()}, Wilcoxon p = {stats.wilcoxon(dx).pvalue:.2g}')
