#!/usr/bin/env python3
"""Variant A of the unified dose-action design: OT-FiLM with --orth-split k_f,k_d (FiLM / dynamics /
physiology on mutually orthogonal learned subspaces, physiology conserved exactly) vs the unconstrained Table 3 FiLM.
Windowed s4 benchmark, N=100, ep6000, same 1100 test patients, scaler pinned to the training cohort (chain_orth.sh).
Rows: Table 3 FiLM (3 seeds), osplit 1-3 (3 seeds), osplit 1-5 (seed 1). Paired: seed-averaged |AUC err| per patient,
osplit 1-3 minus FiLM (same patients, same order -- checked on the true AUC arrays), Wilcoxon."""
import os, sys, json, numpy as np
sys.argv = sys.argv[:1]; sys.path.insert(0, 'scripts/ndm')
from ndm_tables import net_cell, agg, fmt
from scipy.stats import wilcoxon
E3, EO = 'results/s4_pnoise_win/eval', 'results/s4_orth/eval'
nets = (('film', 'FiLM, unconstrained (Table 3)', E3, 'film', (1, 2, 3)),
        ('o13', 'FiLM + orth-split 1/3/6', EO, 'osplit1-3', (1, 2, 3)),
        ('o15', 'FiLM + orth-split 1/5/4 (seed 1 only)', EO, 'osplit1-5', (1,)))
for reg, vis, lab in (('in', 'v1', 'V1 AUC (observed dose)'), ('in', 'v2', 'V2 AUC, new dose 1-8 mg'),
                      ('hi', 'v2', 'V2 AUC, above range 10-12 mg'), ('lo', 'v2', 'V2 AUC, below range 0.25-0.5 mg')):
    print(f'-- {lab}')
    A = {}
    for key, name, d, cfg, seeds in nets:
        A[key] = agg([net_cell(f'{d}/ep006000_{cfg}_n100_s{s}_on_{reg}.json', vis) for s in seeds])
        print(f'   {name:40s} ' + fmt(A[key]))
    if A['film'] is not None and A['o13'] is not None:
        t0 = json.load(open(f'{E3}/ep006000_film_n100_s1_on_{reg}.json'))['per_patient'][f'true_auc_{vis}']
        t1 = json.load(open(f'{EO}/ep006000_osplit1-3_n100_s1_on_{reg}.json'))['per_patient'][f'true_auc_{vis}']
        if np.allclose(t0, t1):
            a, b = A['o13']['abs_mean_over_seeds'], A['film']['abs_mean_over_seeds']
            print(f'   paired (n={len(a)}): orth-split 1/3/6 minus FiLM, median |err| {100 * (np.median(a) - np.median(b)):+.2f} pt; '
                  f'orth-split closer for {100 * np.mean(a < b):.1f}% of patients; Wilcoxon p = {wilcoxon(a, b).pvalue:.2g}')
        else:
            print('   paired comparison skipped: patient order differs')
