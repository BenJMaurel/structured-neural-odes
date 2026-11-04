#!/usr/bin/env python3
"""Clinical baselines on the regimen benchmark of Baaz et al. (main-text Table 3 confirmation sets, review comment 22/10).
MAP-BE: the simulator's structure (one compartment, first-order absorption, linear elimination), population fitted by SAEM on the 200
training patients, MAP on the samples of the training regimen, prediction of every regimen by superposition (mapbe_pop.py, outputs of
scripts/review22/queue_baselines.sh in results/review22/mapbe_regimen/{ctrl,titr0.8}/). Proportional rule: the MAP-BE estimate of the
training-regimen AUC times the ratio of the new amount to the patient's arm amount, defined only for regimens that keep the training
schedule with equal doses (unseen: 350, 600, 850 mg; held-out: 500, 1000 mg). Cells as Table 3: median |AUC(0-30 h) err| % averaged over
the regimens of the column, mean +- s.d. over training sets (sparse randomised: the nine sets of Table 3, seed 14 left out with its pair).
Writes results/review22/regimen_baselines_table.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, glob, json, numpy as np
os.chdir(REPO)
O = 'results/review22/mapbe_regimen'
UNSEEN = ['amt350', 'amt600', 'amt850', 'time400', 'time800']
HELD = ['c_amt500', 'c_amt1000', 'c_load', 'c_missed', 'c_extra']
AMT = {'amt350': 350, 'amt600': 600, 'amt850': 850, 'c_amt500': 500, 'c_amt1000': 1000}
SETS = [('Sparse, randomised', 'ctrl', 'sparse', [5, 6, 7, 8, 9, 10, 11, 12, 13]), ('Sparse, titrated', 'titr0.8', 'sparse', list(range(5, 15))),
        ('Dense, randomised', 'ctrl', 'dense', [3, 4, 5, 6, 7])]
ms = lambda v: f'{np.mean(v):5.1f} +- {np.std(v, ddof=1) if len(v) > 1 else 0:4.1f} (n {len(v)})' if v else '  (none)'
L = [__doc__.split('\n')[0], '', f'{"":20s} {"MAP-BE unseen":>22s} {"MAP-BE held-out":>22s} {"rule unseen (3)":>22s} {"rule held-out (2)":>22s}']
for lab, sub, d, seeds in SETS:
    mu, mh, ru, rh = [], [], [], []
    for s in seeds:
        f = f'{O}/{sub}/N100_{d}_s{s}.json'
        if not os.path.exists(f): continue
        ind = json.load(open(f))['individual']
        if not all(k in ind for k in UNSEEN + HELD): continue
        mu.append(np.mean([ind[k]['auc_abs_median'] for k in UNSEEN])); mh.append(np.mean([ind[k]['auc_abs_median'] for k in HELD]))
        ptr = np.array(ind['train']['auc_pred_per_patient']); arm = np.where(np.arange(len(ptr)) % 2 == 0, 400.0, 800.0)   # test arms alternate
        rule = {}
        for k, a in AMT.items():
            p = np.array(ind[k]['auc_pred_per_patient']); e = np.array(ind[k]['auc_err_per_patient']); true = p / (1 + e)
            rule[k] = 100 * np.median(np.abs(ptr * a / arm / true - 1))
        ru.append(np.mean([rule[k] for k in ('amt350', 'amt600', 'amt850')])); rh.append(np.mean([rule[k] for k in ('c_amt500', 'c_amt1000')]))
    L.append(f'{lab:20s} {ms(mu):>22s} {ms(mh):>22s} {ms(ru):>22s} {ms(rh):>22s}')
L += ['', 'Table 3 reference (dose-event model): sparse randomised 10.9 / 8.2, titrated 12.3 / 12.2, dense 2.6 / 2.2 (unseen / held-out).']
open('results/review22/regimen_baselines_table.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
