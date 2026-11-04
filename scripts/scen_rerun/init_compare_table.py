#!/usr/bin/env python3
"""MAP-BE under three Monolix initialisations (published typical values / Monolix defaults / Monolix auto-init), all
with the combined1 (form-fixed) residual error, vs the latent ODE. Paired on the runs where every method exists and on
the patients every MAP-BE returned. A MAP-BE run counts as FAILED if it aborted (FAILED_*) or returned < MINPAT of the
40 test patients (simulations that crash drop patients); failures are reported per initialisation and excluded from
the paired table. Per run: RMSPE, MPE, median |error| (%). Summaries: median [IQR], mean +- s.d.; paired differences
with Wilcoxon p and the number of runs where the first method is lower. Fit plausibility: Vc_pop 100-2000 L,
KTR_pop 1-10 /h, Vp_pop 50-5000 L, Q_pop 5-1000 L/h.   Usage: init_compare_table.py [scenarios, default "1 2 3"]"""
import os, sys, glob, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__)); from summarise import load
R = 'results/scen_rerun/runs'; MINPAT = int(os.environ.get('MINPAT', 30))
INITS = (('pub', 'published', 'sigc1', 'DONE_c1', 'FAILED_c1_*', 'populationParameters.txt', ''),
         ('def', 'default', 'definit_sigc1', 'DONE_definit_c1', 'FAILED_definit_c1_*', 'populationParameters_definit.txt', '_definit'),
         ('auto', 'auto-init', 'autoinit_sigc1', 'DONE_autoinit', 'FAILED_autoinit_*', 'populationParameters_autoinit.txt', '_autoinit'))
f3 = lambda x: (100 * np.sqrt(np.mean(x ** 2)), 100 * np.mean(x), 100 * np.median(np.abs(x)))
def rd(d, tag):
    x = pd.read_csv(f'{d}/tacro_mapbayest_auc_{tag}.csv'); return pd.Series((x.auc_ipred / x.AUC_observed - 1).values, index=x.ID.values)
def params(d, fname, wsfx):
    p = f'{d}/{fname}'
    if not os.path.exists(p):
        b = os.path.basename(d); i = int(b[1]) * 10000 + int(b[-3:])
        g = glob.glob(f'results/paper_repro/w*/results/{i}{wsfx}/2_test_tacro/populationParameters.txt'); p = g[0] if g else None
    return pd.read_csv(p).set_index('parameter').value if p else None
def plausible(pp):
    return (100 <= pp['Vc_pop'] <= 2000 and 1 <= pp['KTR_pop'] <= 10 and 50 <= pp['Vp_pop'] <= 5000 and 5 <= pp['Q_pop'] <= 1000)
q = lambda v: f'{np.median(v):5.1f} [{np.percentile(v, 25):4.1f}-{np.percentile(v, 75):4.1f}]  {np.mean(v):5.1f} +- {np.std(v, ddof=1):4.1f}'
scens = [int(x) for x in (sys.argv[1].split() if len(sys.argv) > 1 else ['1', '2', '3'])]
for sc in scens:
    dirs = sorted(d for d in glob.glob(f'{R}/s{sc}_seed[0-9][0-9][0-9]') if os.path.exists(f'{d}/DONE'))
    status = {}   # (dir, init) -> 'ok' | 'failed' | 'pending'
    for d in dirs:
        for key, _, tag, done, fpat, _, _ in INITS:
            if os.path.exists(f'{d}/{done}'):
                status[d, key] = 'ok' if len(rd(d, tag)) >= MINPAT else 'failed'
            else:
                status[d, key] = 'failed' if glob.glob(f'{d}/{fpat}') else 'pending'
    print('=' * 110)
    print(f'SCENARIO {sc} (MAP-BE failure = aborted or < {MINPAT}/40 patients returned)')
    for key, lab, *_ in INITS:
        c = {s: sum(status[d, key] == s for d in dirs) for s in ('ok', 'failed', 'pending')}
        print(f'  MAP-BE {lab:10s} init: {c["ok"]:3d} ok, {c["failed"]:2d} failed, {c["pending"]:3d} not run yet')
    ok = [d for d in dirs if all(status[d, k] == 'ok' for k, *_ in INITS)]
    if len(ok) < 3: print('  too few complete runs'); continue
    A = {k: [] for k in ('lode_best', 'lode_final', 'pub', 'def', 'auto')}; pl = {k: [] for k in ('pub', 'def', 'auto')}
    for d in ok:
        b, _ = load(d); S = {k: rd(d, tag) for k, _, tag, *_ in INITS}
        ids = sorted(set(b['lode_best'].index).intersection(*[set(s.index) for s in S.values()]))
        A['lode_best'].append(f3(b['lode_best'].loc[ids].values)); A['lode_final'].append(f3(b['lode_final'].loc[ids].values))
        for k, _, _, _, _, fn, ws in INITS:
            A[k].append(f3(S[k].loc[ids].values)); pp = params(d, fn, ws); pl[k].append(pp is not None and plausible(pp))
    A = {k: np.array(v) for k, v in A.items()}; pl = {k: np.array(v) for k, v in pl.items()}
    print(f'  paired table on the {len(ok)} runs where all three MAP-BE succeeded')
    print(f'  {"":36s} {"RMSPE median [IQR]   mean +- sd":34s} | {"MPE mean +- sd":14s} | med|err| | plausible fits')
    for k, lab in (('lode_best', 'Latent ODE (_best)'), ('lode_final', 'Latent ODE (final epoch)'), ('pub', 'MAP-BE published init + form fix'),
                   ('def', 'MAP-BE default init + form fix'), ('auto', 'MAP-BE auto-init + form fix')):
        extra = f'{pl[k].sum():3d}/{len(ok)}' if k in pl else ''
        print(f'  {lab:36s} {q(A[k][:, 0])} | {A[k][:, 1].mean():+5.2f} +- {A[k][:, 1].std(ddof=1):4.2f} | {np.median(A[k][:, 2]):5.1f}    | {extra}')
    for a, b in (('lode_best', 'auto'), ('lode_final', 'auto'), ('auto', 'pub'), ('auto', 'def')):
        for mi, met in ((0, 'RMSPE'), (2, 'med|err|')):
            dx = A[a][:, mi] - A[b][:, mi]; w = stats.wilcoxon(dx)
            print(f'  {a:10s} - {b:4s} {met:8s}: median diff {np.median(dx):+6.2f}; {a} lower in {(dx < 0).sum():3d}/{len(dx)}; Wilcoxon p = {w.pvalue:.2g}')
    m = pl['auto']
    for name, mm in (('plausible', m), ('degenerate', ~m)):
        if mm.sum() >= 2:
            dx = A['lode_best'][mm, 0] - A['auto'][mm, 0]
            print(f'  [{name:10s} auto-init fits, n={mm.sum():3d}] RMSPE auto-init median {np.median(A["auto"][mm, 0]):5.1f} vs LODE _best '
                  f'{np.median(A["lode_best"][mm, 0]):5.1f}; LODE lower in {(dx < 0).sum()}/{mm.sum()}, Wilcoxon p = {stats.wilcoxon(dx).pvalue:.2g}')
