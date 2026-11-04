#!/usr/bin/env python3
"""Clinical baselines on the two-visit counterfactual task (review comment 22/10), next to the neural rows of main-text Table 2.
Same cohort and test patients as Table 2: confound_vc00_s4_pnoise_win (Michaelis-Menten elimination, hematocrit on Vmax and Km),
100 training patients, 1100 in-range test patients; V1 = observed dose (factual), V2 = alternative dose (counterfactual).
MAP-BE: one MAP fit to the visit-1 samples at 0, 1 and 3 h (the neural models' inputs), prediction of both visits with the
population model of the row (results/s4/ebe/popebe_<tag>_n1100.csv, scripts/monolix/ebe_popmodel*.py):
  true          simulator's own population model (floor: the irreducible error of three noisy samples)
  mlx           simulator's structure and covariate model, SAEM on the 100 training patients
  mlx_l1        Michaelis-Menten, hematocrit on Vmax only (no hematocrit on Km, no Km-Vc correlation)
  mlx_mm_noht   Michaelis-Menten without hematocrit                         (omitted covariate, scenario 2 of Table 1)
  mlx_lin       linear elimination, hematocrit on CL                         (misspecified elimination, scenario 3)
  mlx_lin_noht  linear elimination without hematocrit = Table 1's MAP-BE model (both)
Proportional rule: predicted V2 AUC = factual V1 AUC estimate x d2 / d1, from the best factual estimate available (MAP-BE, true
population model) and from the true V1 AUC (the error that proportionality alone makes). Cells as main-text Table 2 (ndm_tables.metrics):
median |err|, % within +-20 %, RMSPE without the 1 % largest errors, number of patients above 100 %; neural rows mean over seeds.
Writes results/review22/cf_baselines_table.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, glob, numpy as np, pandas as pd
os.chdir(REPO); sys.path.insert(0, 'scripts/ndm')
from ndm_tables import metrics, net_cell, agg

E = 'results/s4/ebe/popebe_{}_n1100.csv'
ROWS = [('MAP-BE, simulator population model (floor)', 'true_pnoise_win'),
        ('MAP-BE, correct model, SAEM', 'mlx_pnoise_win_n100'),
        ('MAP-BE, no hematocrit on Km, SAEM', 'mlx_l1_pnoise_win_n100'),
        ('MAP-BE, omitted hematocrit (MM), SAEM', 'mlx_mm_noht_pnoise_win'),
        ('MAP-BE, linear elimination, SAEM', 'mlx_lin_pnoise_win'),
        ('MAP-BE, linear, no hematocrit (Table 1 model), SAEM', 'mlx_lin_noht_pnoise_win')]
fmt = lambda m1, m2: (f'{m1["med"]:5.1f}  {m1["w20"]:5.1f}   |  {m2["med"]:5.1f}  {m2["w20"]:5.1f}  {m2["trm"]:6.1f}  {m2["n100"]:6.1f}')
L = [__doc__.split('\n')[0], '', f'{"":52s}  V1 med  +-20%  |  V2 med  +-20%  RMSPE99  >100%']
ref = None
for lab, tag in ROWS:
    f = E.format(tag)
    if not os.path.exists(f):
        L.append(f'{lab:52s}  (not run yet)'); continue
    d = pd.read_csv(f).sort_values('ID')
    if ref is None: ref = d
    assert np.allclose(d.auc_true_in.values, ref.auc_true_in.values) and len(d) == 1100, f'{tag}: other test patients'
    L.append(f'{lab:52s}  ' + fmt(metrics(d.auc_true_v1, d.auc_v1), metrics(d.auc_true_in, d.auc_in)))
    if tag == 'mlx_pnoise_win_n100':   # the realistic best factual estimate: MAP-BE with the correct model fitted by SAEM
        L.append(f'{"proportional rule, from this factual estimate":52s}  ' + fmt(metrics(d.auc_true_v1, d.auc_v1), metrics(d.auc_true_in, d.auc_v1 * d.dose_in / d.dose_v1)))
    if tag == 'true_pnoise_win':
        prop = d.auc_v1 * d.dose_in / d.dose_v1
        L.append(f'{"proportional rule, from this factual estimate":52s}  ' + fmt(metrics(d.auc_true_v1, d.auc_v1), metrics(d.auc_true_in, prop)))
        orc = d.auc_true_v1 * d.dose_in / d.dose_v1
        L.append(f'{"proportional rule, from the true V1 AUC":52s}  ' + fmt(metrics(d.auc_true_v1, d.auc_true_v1), metrics(d.auc_true_in, orc)))
L.append('')
# neural rows (main-text Table 2 evaluations); truth set must match the MAP-BE patients
for lab, pat in (('OT-FiLM (Table 2 checkpoints)', ['results/s4_pnoise_win/eval/ep006000_film_n100_s{}_on_in.json'.format(s) for s in (1, 2, 3)]
                  + ['results/s4_pnoise_win_more/eval/ep006000_film_n100_s{}_on_in.json'.format(s) for s in range(4, 11)]),
                 ('OT-FiLM, gamma > 0', [f'results/s4_pnoise_win_gpos/eval/ep006000_film_n100_s{s}_on_in.json' for s in range(1, 11)]),
                 ('dose-conditioned field', ['results/s4_pnoise_win/eval/ep006000_dc_n100_s{}_on_in.json'.format(s) for s in (1, 2, 3)]
                  + ['results/s4_pnoise_win_more/eval/ep006000_dc_n100_s{}_on_in.json'.format(s) for s in range(4, 11)]),
                 ('dose-event model', [f'results/s4_doseh/eval/ep006000_dh7pgkp_osplit1-5_n100_s{s}_on_in.json' for s in (1, 2, 3)])):
    c1, c2 = agg([net_cell(p, 'v1') for p in pat]), agg([net_cell(p, 'v2') for p in pat])
    if c1 is None:
        L.append(f'{lab:52s}  (not available)'); continue
    if ref is not None:
        import json
        tv = np.sort(json.load(open(next(p for p in pat if os.path.exists(p))))['per_patient']['true_auc_v2'])
        assert np.allclose(tv, np.sort(ref.auc_true_in.values), rtol=1e-4), f'{lab}: other test patients than MAP-BE'
    L.append(f'{lab + f" [{c1[chr(110)]} seeds]":52s}  ' + fmt(c1, c2))
open('results/review22/cf_baselines_table.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
