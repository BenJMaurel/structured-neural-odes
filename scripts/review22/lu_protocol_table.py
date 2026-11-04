#!/usr/bin/env python3
"""Job 3 (review 22/10): Lu et al. regimen model (iScience 2021) under the two encoder protocols stored in every run.
'target' = the encoder reads the dosing of the regimen being predicted (Table 3 / Table S5 'as published');
'factual' = the encoder reads the observed regimen and only the doses added during the rollout change -- what their public code does
for a counterfactual regimen (cross-schedule_models/Neural-ODE: run_predict.py -> utils.compute_loss_on_interp(nodosing, test): encoder
on features_o of the observed test data, dosing from test_nodosing.csv).
Cells: unseen = median |AUC err| averaged over the five unseen regimens, mean +- sd over seeds; training regimen; share of test patients
with a predicted unseen-regimen AUC <= 0 (mean over regimens and seeds). Writes results/review22/lu_protocol_table.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import glob, json, os, numpy as np
os.chdir(REPO)
SETS = [('Baaz sparse randomised', 'results/regimen_dh/luis/N100_sparse_s*.json'),
        ('Baaz sparse titrated', 'results/regimen_dh/titr0.8/luis/N100_sparse_s*.json'),
        ('Baaz dense randomised', 'results/regimen_dh/luis/N100_dense_s*.json'),
        ('tacro linear 6', 'results/regimen_dh/tacro_s2/luis/N100_sparse_s*.json'),
        ('tacro linear 12 (visit)', 'results/regimen_dh/tacro_s2/luis/N100_visit_s*.json'),
        ('tacro linear 12 (visit1)', 'results/regimen_dh/tacro_s2/luis/N100_visit1_s*.json'),
        ('tacro linear 61', 'results/regimen_dh/tacro_s2/luis/N100_dense_s*.json'),
        ('tacro saturable 6', 'results/regimen_dh/tacro/luis/N100_sparse_s*.json'),
        ('tacro saturable 61', 'results/regimen_dh/tacro/luis/N100_dense_s*.json')]
L = [__doc__.split('\n')[0], '', f"{'design':26s} {'n':>2s}  {'target unseen':>15s} {'factual unseen':>15s} {'training':>9s}  AUC<=0 target / factual"]
for name, pat in SETS:
    fs = sorted(f for f in glob.glob(pat) if 'protocol' not in f)
    rows = []
    for f in fs:
        d = json.load(open(f))
        if d.get('protocol', 'cf') != 'cf' or 'individual_factual_encoder' not in d: continue
        un = [k for k in d['individual'] if k != 'train']
        u_t = np.mean([d['individual'][k]['auc_abs_median'] for k in un])
        u_f = np.mean([d['individual_factual_encoder'][k]['auc_abs_median'] for k in un])
        neg = lambda key: np.mean([np.mean(np.array(d[key][k]['auc_pred_per_patient']) <= 0) for k in un]) if 'auc_pred_per_patient' in d[key][un[0]] else np.nan
        rows.append((u_t, u_f, d['individual']['train']['auc_abs_median'], neg('individual'), neg('individual_factual_encoder')))
    if not rows:
        L.append(f'{name:26s}  no runs'); continue
    a = np.array(rows); ms = lambda c: f'{a[:, c].mean():5.1f} +- {a[:, c].std(ddof=1) if len(a) > 1 else 0:4.1f}'
    L.append(f'{name:26s} {len(a):2d}  {ms(0):>15s} {ms(1):>15s} {a[:, 2].mean():9.1f}  {100 * np.nanmean(a[:, 3]):4.0f} % / {100 * np.nanmean(a[:, 4]):4.0f} %')
open('results/review22/lu_protocol_table.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
