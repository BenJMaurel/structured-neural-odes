#!/usr/bin/env python3
"""Latent-size sweep of the Table 3 FiLM (queue_latent.sh): -l 6..20 step 2 on windowed pnoise N=100,
ep6000, 1100 test patients, scaler pinned. l = 10 is the existing FiLM (seeds 1-3 in the Table 3 dir, 4-10 in
results/s4_seeds); the other sizes have 3 seeds each in results/s4_latent. Uses whatever seeds exist. In range first
(V1, V2 1-8 mg), then out of range. Seed level: Mann-Whitney of each size's seed medians vs l = 10."""
import os, sys, numpy as np
sys.argv = sys.argv[:1]; sys.path.insert(0, 'scripts/ndm')
from ndm_tables import net_cell, agg, fmt
from scipy.stats import mannwhitneyu
E3, ES, EL = 'results/s4_pnoise_win/eval', 'results/s4_seeds/eval', 'results/s4_latent/eval'


def path(L, s, reg):
    if L == 10:
        return f'{E3 if s <= 3 else ES}/ep006000_film_n100_s{s}_on_{reg}.json'
    return f'{EL}/ep006000_film_l{L}_n100_s{s}_on_{reg}.json'


SIZES = (6, 8, 10, 12, 14, 16, 18, 20)
for reg, vis, lab in (('in', 'v1', 'V1 AUC (observed dose), in range'), ('in', 'v2', 'V2 AUC, new dose 1-8 mg, in range'),
                      ('hi', 'v2', 'V2 AUC, above range 10-12 mg'), ('lo', 'v2', 'V2 AUC, below range 0.25-0.5 mg')):
    print(f'== {lab}')
    med = {}
    for L in SIZES:
        seeds = [s for s in range(1, 11) if os.path.exists(path(L, s, reg))]
        cells = [net_cell(path(L, s, reg), vis) for s in seeds]
        med[L] = np.array([c['med'] for c in cells])
        print(f'   latent {L:2d} ' + fmt(agg(cells), show_cov=(vis == 'v2')))
    ref = med[10]
    line = []
    for L in SIZES:
        if L == 10 or len(med[L]) == 0 or len(ref) < 2:
            continue
        p = mannwhitneyu(med[L], ref, alternative='two-sided').pvalue if len(med[L]) >= 2 else float('nan')
        line.append(f'l{L} {med[L].mean() - ref.mean():+.2f} (p {p:.2g})')
    print('   mean of seed medians minus l=10: ' + '; '.join(line))
