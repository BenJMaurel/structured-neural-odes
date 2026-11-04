#!/usr/bin/env python3
"""Variant B of the unified dose-action design: --orth-split 1,5 --dose-history 7 (no FiLM transport; each
visit simulated through its 7-dose history from the drug-free state of the encoded physiology) vs the unconstrained Table 3
FiLM and variant A (FiLM + orth-split 1,5). Two B versions: 'dh7' (first run, 24 h intervals for EVERY patient -- wrong for
the ~40 % Prograf patients, who are dosed every 12 h) and 'dh7pg' (--dose-interval-prograf 12: per-formulation intervals,
chain_dosehpg.sh). Windowed s4, N=100, ep6000, same 1100 test patients, scaler pinned. Uses whatever seeds 1-10 exist
(FiLM 1-3 in the Table 3 dir, 4-10 in results/s4_seeds). Paired: seed-averaged |AUC err| per patient vs FiLM, Wilcoxon."""
import os, sys, json, numpy as np
sys.argv = sys.argv[:1]; sys.path.insert(0, 'scripts/ndm')
from ndm_tables import net_cell, agg, fmt
from scipy.stats import wilcoxon
E3, ES, EA, EB = 'results/s4_pnoise_win/eval', 'results/s4_seeds/eval', 'results/s4_orth/eval', 'results/s4_doseh/eval'
path = {'film': lambda s, r: f'{E3 if s <= 3 else ES}/ep006000_film_n100_s{s}_on_{r}.json',
        'a15': lambda s, r: f'{EA}/ep006000_osplit1-5_n100_s{s}_on_{r}.json',
        'b': lambda s, r: f'{EB}/ep006000_dh7_osplit1-5_n100_s{s}_on_{r}.json',
        'bpg': lambda s, r: f'{EB}/ep006000_dh7pg_osplit1-5_n100_s{s}_on_{r}.json'}
name = {'film': 'FiLM, unconstrained (Table 3)', 'a15': 'A: FiLM + orth-split 1/5/4',
        'b': 'B: dose history, 24 h for all (wrong)', 'bpg': 'B fixed: 12 h Prograf / 24 h Advagraf'}
for reg, vis, lab in (('in', 'v1', 'V1 AUC (observed dose)'), ('in', 'v2', 'V2 AUC, new dose 1-8 mg'),
                      ('hi', 'v2', 'V2 AUC, above range 10-12 mg'), ('lo', 'v2', 'V2 AUC, below range 0.25-0.5 mg')):
    print(f'-- {lab}')
    A, S = {}, {}
    for k in path:
        S[k] = [s for s in range(1, 11) if os.path.exists(path[k](s, reg))]
        A[k] = agg([net_cell(path[k](s, reg), vis) for s in S[k]])
        print(f'   {name[k]:40s} ' + fmt(A[k], show_cov=(vis == 'v2')))
    if A['film'] is None:
        continue
    t0 = json.load(open(path['film'](S['film'][0], reg)))['per_patient'][f'true_auc_{vis}']
    for k in ('a15', 'b', 'bpg'):
        if A[k] is None:
            continue
        if not np.allclose(t0, json.load(open(path[k](S[k][0], reg)))['per_patient'][f'true_auc_{vis}']):
            print(f'   {k}: paired comparison skipped, patient order differs'); continue
        a, b = A[k]['abs_mean_over_seeds'], A['film']['abs_mean_over_seeds']
        print(f'   paired {k} (seeds {S[k]}) minus FiLM (seeds {S["film"]}): median |err| {100 * (np.median(a) - np.median(b)):+.2f} pt; '
              f'closer for {100 * np.mean(a < b):.1f}% of patients; Wilcoxon p = {wilcoxon(a, b).pvalue:.2g}')
