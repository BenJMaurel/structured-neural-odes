#!/usr/bin/env python3
"""Dose-cond tuning on Baaz et al.'s linear regimen benchmark, sparse seeds 0-4 (queue_dc_tune.sh; ).
Configs: default (std-0.1 init, 6000 it; results/regimen_dh/dosecond_n), z6 (zero-init output layer, 6000 it), z12 (zero-init,
12000 it), d12 (default init, 12000 it). All: --dose-cond --enc-dose norm --obsrv-mgl 20.
SELECTION RULE, declared before the runs: lowest mean TRAINING-REGIMEN median |AUC err| (the observed regimen of the test patients,
a factual fit) over the 5 seeds; within 1 point, the cheaper config (6000 before 12000 it, zero-init counted as free). The
unseen-regimen error is printed for information and is NOT used to choose. Reference rows: dose-event model (Bfixkp20n) and
EB-VAE (Table 4, same seeds)."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, json, numpy as np
R = 'results/regimen_dh'; EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'
CONFIGS = (('default', f'{R}/dosecond_n', 6000, 0), ('z6', f'{R}/dc_tune/z6', 6000, 1), ('z12', f'{R}/dc_tune/z12', 12000, 2),
           ('d12', f'{R}/dc_tune/d12', 12000, 3))
SEEDS = range(5)


def ebpath(s):   # Table 4's EB-VAE runs (table4.py)
    p = f'{EB}/res_titr/ctrl/N100_sparse_s{s}.json'
    return p if os.path.exists(p) else f'{EB}/res/N100_sparse' + (f'_s{s}' if s else '') + '.json'


def row(lab, d, ref=False):
    path = d if callable(d) else (lambda s: f'{d}/N100_sparse_s{s}.json')
    js = {s: json.load(open(path(s))) for s in SEEDS if os.path.exists(path(s))}
    if not js:
        print(f'  {lab:22s} (no runs)'); return None
    tr = np.array([j['individual']['train']['auc_abs_median'] for j in js.values()])
    un = np.array([np.mean([v['auc_abs_median'] for k, v in j['individual'].items() if k != 'train']) for j in js.values()])
    tau = [round(j.get('dose_cond_tau_h', np.nan), 2) for j in js.values()]
    print(f'  {lab:22s} n {len(js)}  training regimen {tr.mean():5.1f} +- {tr.std(ddof=1) if len(tr) > 1 else 0:4.1f} {np.round(tr, 1).tolist()}'
          f'  | unseen {un.mean():5.1f} +- {un.std(ddof=1) if len(un) > 1 else 0:4.1f} {np.round(un, 1).tolist()}'
          + ('' if ref else f'  | tau {tau}  | seeds with training-regimen error > 25 %: {int((tr > 25).sum())}'))
    return tr.mean(), len(js)


print('Dose-cond tuning, Baaz linear simulator, N100 sparse (median |AUC 0-30 h err| %, mean +- sd over seeds)')
res = {}
for lab, d, it, cost in CONFIGS:
    r = row(f'{lab} ({it} it)', d)
    if r and r[1] == 5:
        res[lab] = (r[0], cost)
print('  -- references')
row('dose-event (Bfixkp20n)', f'{R}/Bfixkp20n', ref=True)
row('EB-VAE', ebpath, ref=True)
if res:
    best = min(v[0] for v in res.values())
    sel = min((k for k, v in res.items() if v[0] <= best + 1.0), key=lambda k: res[k][1])
    print(f'\nSELECTED (rule above): {sel}  (mean training-regimen error {res[sel][0]:.1f}; best {best:.1f})')
    json.dump(dict(selected=sel, training_regimen_means={k: v[0] for k, v in res.items()}), open(f'{R}/dc_tune/selected.json', 'w'), indent=1)
