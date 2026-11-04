#!/usr/bin/env python3
"""Baaz's regimen test: EB-VAE (their public pipeline, third_party/baaz_ebvae/ours/regimen/res) vs our model (B = dose
events in the drug subspace, physiology conserved; plain = same latent ODE without the B structure), same patients
(checked by dh_regimen.py), B2 = B version 2 (dh_regimen_v2.py: fixed noise, linear compartments), mapbe_oracle = MAP-BE with the TRUE model and population parameters (mapbe_regimen.py; the ceiling for population PK), same metrics (run_regimen.py): median individual |AUC(0-30 h) err| % (bias), 400 test
patients encoded from their training-regimen data, predicted under the training regimen and 5 unseen ones. Handoff
Seeds: EB-VAE seed 0 for all 6 configurations + seeds 1-2 for the sparse ones; ours the same. Uses whatever exists."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, json, numpy as np
EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen/res'
OURS = 'results/regimen_dh'
REGS = ('train', 'amt350', 'amt600', 'amt850', 'time400', 'time800')


def load(model, n, d, s):
    p = (f'{EB}/N{n}_{d}' + (f'_s{s}' if s else '') + '.json') if model == 'ebvae' else f'{OURS}/{model}/N{n}_{d}_s{s}.json'
    return json.load(open(p)) if os.path.exists(p) else None


print('median individual |AUC(0-30 h) error| % (median bias %) per regimen; 400 test patients; train = factual regimen')
print(f'{"config":16s} {"model":8s} ' + ' '.join(f'{r:>13s}' for r in REGS) + '   unseen mean')
for d in ('dense', 'sparse'):
    for n in (25, 100, 250):
        for s in ((0,) if d == 'dense' else (0, 1, 2)):
            for m in ('mapbe_oracle', 'ebvae', 'B', 'B2', 'plain'):
                r = load(m, n, d, s)
                if r is None:
                    continue
                cells = [f'{r["individual"][g]["auc_abs_median"]:5.1f} ({r["individual"][g]["auc_bias_median"]:+5.1f})' for g in REGS]
                unseen = np.mean([r['individual'][g]['auc_abs_median'] for g in REGS[1:]])
                print(f'N{n:<3d} {d:6s} s{s}   {m:8s} ' + ' '.join(f'{c:>13s}' for c in cells) + f'   {unseen:5.1f}')
        print()
print('seed means (sparse: seeds 0-2 where both models have them) -- unseen-regimen mean of the median |AUC err|:')
for d in ('dense', 'sparse'):
    for n in (25, 100, 250):
        row = []
        for m in ('mapbe_oracle', 'ebvae', 'B', 'B2', 'plain'):
            v = [np.mean([r['individual'][g]['auc_abs_median'] for g in REGS[1:]])
                 for s in ((0,) if d == 'dense' else (0, 1, 2)) for r in [load(m, n, d, s)] if r is not None]
            row.append(f'{m} {np.mean(v):5.1f} (n={len(v)})' if v else f'{m}   n/a')
        print(f'   N{n:<3d} {d:6s}  ' + '   '.join(row))
print('\npopulation (prior draws vs true population): mean of the q10/q50/q90 relative errors %, per regimen')
for d in ('dense', 'sparse'):
    for n in (25, 100, 250):
        for m in ('mapbe_oracle', 'ebvae', 'B', 'B2', 'plain'):
            r = load(m, n, d, 0)
            if r is None or 'population' not in r:
                continue
            print(f'   N{n:<3d} {d:6s} {m:8s} ' + ' '.join(f'{g} {np.mean(list(r["population"][g].values())):5.1f}' for g in REGS))
