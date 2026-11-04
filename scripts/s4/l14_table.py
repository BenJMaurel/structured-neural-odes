#!/usr/bin/env python3
"""Latent-14 study (queue_l14.sh): variant A splits at -l 14 vs FiLM at -l 14 and -l 10, plus the plain
latent ODE (paper model, linear decoder) on the same windowed cohort at -l 10 / 14 (V1 only: it cannot transport a dose).
Windowed pnoise N=100, ep6000 (FiLM / A) or final (plain), 1100 test patients. In range first. Seed level first (carries
seed variance), then paired per patient where the patients can be matched. Uses whatever seeds exist."""
import os, sys, json, numpy as np
sys.argv = sys.argv[:1]; sys.path.insert(0, 'scripts/ndm')
from ndm_tables import net_cell, lode_cell, agg, fmt
from scipy.stats import mannwhitneyu, wilcoxon
E3, ES, EA, EL = 'results/s4_pnoise_win/eval', 'results/s4_seeds/eval', 'results/s4_orth/eval', 'results/s4_latent/eval'
nets = {'FiLM l10': lambda s, r: f'{E3 if s <= 3 else ES}/ep006000_film_n100_s{s}_on_{r}.json',
        'FiLM l14': lambda s, r: f'{EL}/ep006000_film_l14_n100_s{s}_on_{r}.json'}
for K in (5, 7):
    nets[f'A l10 1,{K} (phys {10 - 1 - K})'] = (lambda K: lambda s, r: f'{EA}/ep006000_osplit1-{K}_n100_s{s}_on_{r}.json')(K)
for K in (5, 7, 9, 11):
    nets[f'A l14 1,{K} (phys {14 - 1 - K})'] = (lambda K: lambda s, r: f'{EL}/ep006000_osplit1-{K}_l14_n100_s{s}_on_{r}.json')(K)
seeds = lambda k, r: [s for s in range(1, 11) if os.path.exists(nets[k](s, r))]

for reg, vis, lab in (('in', 'v1', 'V1 AUC (observed dose), in range'), ('in', 'v2', 'V2 AUC, new dose 1-8 mg, in range'),
                      ('hi', 'v2', 'V2 AUC, above range 10-12 mg'), ('lo', 'v2', 'V2 AUC, below range 0.25-0.5 mg')):
    print(f'== {lab}')
    med = {}
    for k in nets:
        cells = [net_cell(nets[k](s, reg), vis) for s in seeds(k, reg)]
        med[k] = np.array([c['med'] for c in cells])
        print(f'   {k:24s} ' + fmt(agg(cells), show_cov=(vis == 'v2')))
    if reg == 'in' and vis == 'v1':
        for L in (10, 14):
            for conv, sfx in (('harness', '_harness'), ('paper', '')):
                cells = [lode_cell(f'results/lode_s4pnwin_l{L}/eval/s{s}_final{sfx}.json') for s in (1, 2, 3)]
                print(f'   {"plain LODE l" + str(L) + " (" + conv + ")":24s} ' + fmt(agg(cells)))
    for ref in ('FiLM l14', 'FiLM l10'):
        if len(med[ref]) < 2:
            continue
        line = []
        for k in nets:
            if k.startswith('A l14') and len(med[k]) >= 2:
                line.append(f'{k.split(" (")[0][6:]} {med[k].mean() - med[ref].mean():+.2f} (p {mannwhitneyu(med[k], med[ref], alternative="two-sided").pvalue:.2g})')
        if k and line:
            print(f'   A l14 split minus {ref} (mean of seed medians, Mann-Whitney): ' + '; '.join(line))
    if len(med['FiLM l14']) >= 2 and len(med['FiLM l10']) >= 2:
        print(f'   FiLM l14 minus FiLM l10: {med["FiLM l14"].mean() - med["FiLM l10"].mean():+.2f} '
              f'(p {mannwhitneyu(med["FiLM l14"], med["FiLM l10"], alternative="two-sided").pvalue:.2g})')

# paired, V1: FiLM (seed-averaged |err|) vs plain LODE (seed-averaged), patients matched on their true AUC
print('\n== paired V1, FiLM vs plain latent ODE (harness convention), patients matched on the true AUC')
for L in (10, 14):
    lc = [lode_cell(f'results/lode_s4pnwin_l{L}/eval/s{s}_final_harness.json') for s in (1, 2, 3)]
    lc = [c for c in lc if c is not None]
    if not lc:
        continue
    j = json.load(open(f'results/lode_s4pnwin_l{L}/eval/s1_final_harness.json'))['per_series']
    v1 = np.array([i.endswith('_1') for i in j['id']]); t_lode = np.array(j['true_auc'])[v1]
    a_lode = np.mean([c['abs'] for c in lc], 0)
    for fk in ('FiLM l10', 'FiLM l14'):
        ss = seeds(fk, 'in')
        if not ss:
            continue
        t_film = np.array(json.load(open(nets[fk](ss[0], 'in')))['per_patient']['true_auc_v1'])
        a_film = np.mean([net_cell(nets[fk](s, 'in'), 'v1')['abs'] for s in ss], 0)
        o1, o2 = np.argsort(t_film), np.argsort(t_lode)
        if len(o1) == len(o2) and np.allclose(t_film[o1], t_lode[o2], rtol=1e-4):
            a, b = a_film[o1], a_lode[o2]
            print(f'   {fk} (seeds {ss}) minus plain LODE l{L} (seeds {len(lc)}): median |err| {100 * (np.median(a) - np.median(b)):+.2f} pt; '
                  f'FiLM closer for {100 * np.mean(a < b):.1f}%; Wilcoxon p = {wilcoxon(a, b).pvalue:.2g}')
        else:
            print(f'   {fk} vs plain LODE l{L}: patients could not be matched on the true AUC -- seed-level comparison only')
