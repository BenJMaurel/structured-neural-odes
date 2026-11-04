#!/usr/bin/env python3
"""Note S6 'Dose response by observed arm' over ALL training sets: merge development and new runs): development runs
(sparse randomised s0-4, titrated s0-2, dense s0-2) + confirmation runs (randomised s5-13 -- s14 left out with its pair, the EB-VAE
diverged --, titrated s5-14, dense s3-7). Definitions of scripts/regimen/baaz_arm_split.py: within-patient ratio / true (median over
patients), own-regimen median signed error by observed arm (even test index 400 mg, odd 800 mg), cross cells 800->350 and 400->850,
unseen = mean of the median |AUC err| over the five original unseen regimens. Writes results/regimen_dh/arm_split_all.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json
import numpy as np
from scipy.stats import spearmanr
os.chdir(REPO)
EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'; R = 'results/regimen_dh'
sys.path.insert(0, EB); import gen_regimens as G
UNSEEN = ('amt350', 'amt600', 'amt850', 'time400', 'time800')
SETS = {'randomised': [('sparse', 'ctrl', s) for s in list(range(5)) + list(range(5, 14))],
        'titrated': [('sparse', 'titr0.8', s) for s in list(range(3)) + list(range(5, 15))],
        'dense': [('dense', 'ctrl', s) for s in list(range(3)) + list(range(3, 8))]}


def path(m, d, sub, s):
    new = s >= 5 or (d == 'dense' and s >= 3)
    tag = f'N100_{d}_s{s}.json'
    if m == 'eb':
        return f'{EB}/res_titr/confirm/{sub}/{tag}' if new else f'{EB}/res_titr/{sub}/{tag}'
    return f'{R}/confirm/{sub}/Bfixkp20n/{tag}' if new else f'{R}/{"titr0.8/" if sub == "titr0.8" else ""}Bfixkp20n/{tag}'


def arm_ratio(sub, s):
    P = G.draw(200, np.random.default_rng(s))
    arm = G.titrate(P, 0.8, s)[0] if sub == 'titr0.8' else np.where(np.arange(200) % 2 == 0, 400.0, 800.0)
    t, c = G.simulate(P, [0, 8, 16, 24], np.repeat(arm[:, None], 4, 1)); a = np.trapezoid(c, t, axis=1)
    return float(np.exp(np.mean(np.log(a[arm == 800])) - np.mean(np.log(a[arm == 400]))))


L = [__doc__.split('\n')[0], '']
for name, sets in SETS.items():
    rows = {'eb': [], 'ours': []}
    for d, sub, s in sets:
        t = json.load(open(path('eb', d, sub, s)))['individual']
        tru = {g: np.array(t[g]['auc_pred_per_patient']) / (1 + np.array(t[g]['auc_err_per_patient'])) for g in ('amt350', 'amt850')}
        for m in rows:
            r = json.load(open(path(m, d, sub, s)))['individual']
            e = {g: np.array(r[g]['auc_err_per_patient']) for g in ('train', 'amt350', 'amt850')}
            q = (1 + e['amt850']) / (1 + e['amt350'])
            rows[m].append(dict(s=s, rel=float(np.median(q)), unseen=float(np.mean([r[g]['auc_abs_median'] for g in UNSEEN])),
                                own400=100 * np.median(e['train'][0::2]), own800=100 * np.median(e['train'][1::2]),
                                c83=100 * np.median(e['amt350'][1::2]), c48=100 * np.median(e['amt850'][0::2]),
                                ar=arm_ratio(sub, s) if d == 'sparse' else np.nan))
    L.append(f'=== {name}: {len(sets)} training sets')
    for m, lab in (('eb', 'EB-VAE'), ('ours', 'dose-event')):
        X = rows[m]; f = lambda k: np.mean([x[k] for x in X])
        worst = max(X, key=lambda x: x['unseen']); flat = min(X, key=lambda x: x['rel'])
        line = (f'   {lab:10s} rel {f("rel"):.2f}  own 400|800 {f("own400"):+.1f}|{f("own800"):+.1f}  800->350 {f("c83"):+.1f}  '
                f'400->850 {f("c48"):+.1f}  least accurate run s{worst["s"]} ({worst["unseen"]:.1f} %, rel {worst["rel"]:.2f}); '
                f'flattest s{flat["s"]} (rel {flat["rel"]:.2f}){"  SAME" if worst is flat else ""}')
        if name != 'dense':
            rho = spearmanr([x['ar'] for x in X], [x['rel'] for x in X]).correlation
            line += f'  rank corr(arm ratio, rel) {rho:.2f}'
        L.append(line)
    L.append('')
open(f'{R}/arm_split_all.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
