#!/usr/bin/env python3
"""Trough-only latent ODE (chain_trough.sh) vs MAP-BE with the TRUE population model under the same sampling
(oracle_trough.py), windowed s4 benchmark, 1100 test patients. Rows: networks (median |AUC err| etc., mean over 3 seeds),
MAP-BE trough-only (genotype known / unknown Gaussian / unknown true mixture), and the 3-sample references (ours =
Table 3 FiLM; MAP-BE true model). Paired: seed-averaged network |err| vs MAP-BE (Gaussian, genotype unknown), patients
matched on true AUC (checked one-to-one). V1 = observed dose; V2 in = new dose 1-8 mg; hi/lo networks only."""
import os, sys, json, numpy as np, pandas as pd
sys.argv = sys.argv[:1]; sys.path.insert(0, 'scripts/ndm')
from ndm_tables import metrics, net_cell, agg
from scipy.stats import wilcoxon
E, E3 = 'results/s4_trough/eval', 'results/s4_pnoise_win/eval'
O = pd.read_csv('results/s4/oracle_trough/oracle_trough_n1100.csv')
fmt = lambda m: (f'med {m["med"]:5.1f}  bias {m["bias"]:+6.1f}  w20 {m["w20"]:5.1f}  trm {m["trm"]:6.1f}  n>100% {m["n100"]:5.1f}'
                 + (f'   [seeds {" ".join(f"{x:.1f}" for x in m["med_seeds"])}]' if 'med_seeds' in m else ''))
nets = (('trough_hidecyp', 'Latent ODE, trough only, genotype UNMEASURED', E, 'trough_hidecyp'),
        ('trough', 'Latent ODE, trough only, genotype given', E, 'trough'),
        ('film3', 'Latent ODE, 3 samples, genotype given (Table 3)', E3, 'film'))
for reg, vis, lab in (('in', 'v1', 'V1 AUC (observed dose)'), ('in', 'v2', 'V2 AUC, new dose 1-8 mg'),
                      ('hi', 'v2', 'V2 AUC, above range 10-12 mg'), ('lo', 'v2', 'V2 AUC, below range 0.25-0.5 mg')):
    print(f'-- {lab}')
    A = {}
    for key, name, d, cfg in nets:
        A[key] = agg([net_cell(f'{d}/ep006000_{cfg}_n100_s{s}_on_{reg}.json', vis) for s in (1, 2, 3)])
        print(f'   {name:52s} ' + (fmt(A[key]) if A[key] else 'n/a'))
    if reg == 'in':
        tgt = 'v1' if vis == 'v1' else 'in'
        for col, name in (('trough_known', 'MAP-BE true model, trough only, genotype known'),
                          ('trough_gauss', 'MAP-BE true model, trough only, genotype UNMEASURED (Gaussian)'),
                          ('trough_mix_mean', 'MAP-BE true model, trough only, UNMEASURED (true mixture)'),
                          ('three_known', 'MAP-BE true model, 3 samples, genotype known')):
            print(f'   {name:52s} ' + fmt(metrics(O[f'auc_true_{tgt}'], O[f'{col}_{tgt}'])))
        a = A.get('trough_hidecyp')
        if a is not None:
            j = json.load(open(f'{E}/ep006000_trough_hidecyp_n100_s1_on_in.json'))['per_patient']
            t_net = np.array(j[f'true_auc_{vis}']); t_or = O[f'auc_true_{tgt}'].values
            order = np.argsort(t_or); pos = np.searchsorted(t_or[order], t_net).clip(0, len(t_or) - 1)
            idx = order[pos]; ok = np.isclose(t_or[idx], t_net, rtol=1e-4)
            if ok.mean() > 0.99 and len(set(idx[ok])) == ok.sum():
                dn = a['abs_mean_over_seeds'][ok]; do = (O[f'trough_gauss_{tgt}'] / O[f'auc_true_{tgt}'] - 1).abs().values[idx[ok]]
                print(f'   paired (n={ok.sum()}): network (genotype unmeasured) minus MAP-BE (Gaussian, unmeasured), median |err| '
                      f'{100 * (np.median(dn) - np.median(do)):+.2f} pt; network closer for {100 * np.mean(dn < do):.1f}% of patients; '
                      f'Wilcoxon p = {wilcoxon(dn, do).pvalue:.2g}')
            else:
                print(f'   paired comparison skipped: true-AUC matching not one-to-one ({100 * ok.mean():.1f}% matched)')
