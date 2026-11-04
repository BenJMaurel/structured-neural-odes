#!/usr/bin/env python3
"""Fairness checks for the Lu comparison (queue_fair.sh). N100 sparse seeds 0-2 / dense seed 0.
A. Lu iScience tuning sweep on Baaz's counterfactual test (lr x weight decay, + their own optimiser: batch 1, lr 5e-5, L2 0.1):
   mean over the 5 unseen regimens of the median |AUC(0-30 h) err| %, as published (target regimen in the encoder) and with
   the factual encoder; factual-regimen error; test-1 identity ratio (mean over the 5 unseen regimens). The best cell is
   selected ON THE TEST METRIC: an optimistic bound for Lu, compared with our untuned control runs.
B. The iScience format for every model (encoder sees the first dosing interval, predict the rest of the same regimen):
   median |AUC(t2-30 h) err| %, t2 = second dose (8 h; 3 h for time400/800); unseen = mean over the 5 regimens not in training."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, glob, json, numpy as np
R = 'results/regimen_dh'; EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'
CONFIGS = (('sparse', (0, 1, 2)), ('dense', (0,)))
REGS = ('train', 'amt350', 'amt600', 'amt850', 'time400', 'time800'); UNSEEN = REGS[1:]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def unseen(j, key='individual'):
    return float(np.mean([j[key][g]['auc_abs_median'] for g in UNSEEN]))


print('A. Lu iScience tuning sweep, Baaz counterfactual test: unseen-regimen error % as published | factual encoder '
      '| factual regimen | identity ratio; mean over seeds (n)')
cfgs = ['default (lr1e-3_wd0)'] + sorted(os.path.basename(d) for d in glob.glob(f'{R}/luis_sweep/*'))
for d, seeds in CONFIGS:
    print(f'-- N100 {d}, seeds {list(seeds)}')
    rows = []
    for c in cfgs:
        base = f'{R}/luis' if c.startswith('default') else f'{R}/luis_sweep/{c}'
        js = [j for s in seeds for j in [load(f'{base}/N100_{d}_s{s}.json')] if j is not None]
        if not js:
            continue
        a = np.mean([unseen(j) for j in js]); b = np.mean([unseen(j, 'individual_factual_encoder') for j in js])
        f = np.mean([j['individual']['train']['auc_abs_median'] for j in js])
        idr = np.mean([np.mean([j['identity'][g] for g in UNSEEN]) for j in js])
        rows.append((c, a, b, f, idr, len(js), [round(unseen(j), 1) for j in js]))
    best_a = min(r[1] for r in rows) if rows else None; best_b = min(r[2] for r in rows) if rows else None
    for c, a, b, f, idr, n, per in rows:
        mark = (' <- best as published' if a == best_a else '') + (' <- best factual encoder' if b == best_b else '')
        print(f'   {c:24s} {a:6.1f} | {b:6.1f} | {f:5.1f} | {idr:4.2f}  (n={n}; seeds {per}){mark}')
ctrl = {'sparse': {'EB-VAE': 19.7, 'B': 25.6, 'B2': 14.4}, 'dense': {'EB-VAE': 14.1, 'B': 18.6, 'B2': 4.1}}
print('   reference (untuned, same test): ' + '; '.join(f'{d}: ' + ', '.join(f'{k} {v}' for k, v in m.items()) for d, m in ctrl.items()))

print('\nB. iScience format (first dosing interval observed, rest of the same regimen predicted): median |AUC(t2-30 h) err| %')
MODELS = (('mapbe_oracle', 'MAP-BE oracle (ceiling)'), ('luis', 'Lu iScience'), ('lunmi', 'Lu NMI'), ('ebvae', 'EB-VAE'),
          ('B', 'B (ours)'), ('B2', 'B2 (ours)'), ('B2_encnorm', 'B2, enc. dose-normalised'), ('B2_encnone', 'B2, enc. no dose value'))
for d, seeds in CONFIGS:
    print(f'-- N100 {d}, seeds {list(seeds)}      ' + ' '.join(f'{g:>8s}' for g in REGS) + '   unseen  (new amounts / new times)')
    for m, lab in MODELS:
        js = [j for s in seeds for j in [load(f'{EB}/res_firstint/N100_{d}_s{s}.json' if m == 'ebvae' else f'{R}/firstint/{m}/N100_{d}_s{s}.json')]
              if j is not None]
        if not js:
            print(f'   {lab:22s} n/a'); continue
        cell = {g: np.mean([j['individual'][g]['auc_abs_median'] for j in js]) for g in REGS}
        amt = np.mean([cell[g] for g in ('amt350', 'amt600', 'amt850')]); tim = np.mean([cell[g] for g in ('time400', 'time800')])
        print(f'   {lab:22s} (n={len(js)}) ' + ' '.join(f'{cell[g]:8.1f}' for g in REGS) +
              f'   {np.mean([cell[g] for g in UNSEEN]):6.1f}  ({amt:.1f} / {tim:.1f})   seeds {[round(unseen(j), 1) for j in js]}')
