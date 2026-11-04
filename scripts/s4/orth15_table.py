#!/usr/bin/env python3
"""Is variant A (--orth-split 1,5) better than the unconstrained Table 3 FiLM IN RANGE? . Seed 1 alone was 0.5-0.8 pt lower on median
|AUC err|, within seed noise. A seeds 2-3: chain_orth15_seeds.sh; A seeds 4-6 and FiLM seeds 4-6: chain_seeds456.sh
(FiLM 4-6 live in results/s4_seeds/, NOT in the Table 3 directory). Variant B (--dose-history 7, shown for
reference. Windowed s4, N=100, ep6000, same 1100 test patients, scaler pinned. Uses whatever seeds 1-10 exist (FiLM and A seeds 7-10: queue_night.sh,. Seed-level
evidence first (it is the one that carries seed variance), then paired per-patient comparisons on the same patients."""
import os, sys, json, numpy as np
sys.argv = sys.argv[:1]; sys.path.insert(0, 'scripts/ndm')
from ndm_tables import net_cell, agg, fmt
from scipy.stats import wilcoxon, mannwhitneyu
rng = np.random.default_rng(0)
E3, ES, EA, EB = 'results/s4_pnoise_win/eval', 'results/s4_seeds/eval', 'results/s4_orth/eval', 'results/s4_doseh/eval'
path = {'film': lambda s: f'{E3 if s <= 3 else ES}/ep006000_film_n100_s{s}_on_in.json',
        'a15': lambda s: f'{EA}/ep006000_osplit1-5_n100_s{s}_on_in.json',
        'b': lambda s: f'{EB}/ep006000_dh7_osplit1-5_n100_s{s}_on_in.json'}
name = {'film': 'FiLM, unconstrained (Table 3)', 'a15': 'A: FiLM + orth-split 1/5/4', 'b': 'B: dose history 7 (ref.)'}
seeds = {k: [s for s in range(1, 11) if os.path.exists(path[k](s))] for k in path}


def boot_ci(da, db, n_boot=1000):
    n = len(da)
    b = [100 * (np.median(da[i]) - np.median(db[i])) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return np.percentile(b, [2.5, 97.5])


print(f'seeds found: FiLM {seeds["film"]}, A {seeds["a15"]}, B {seeds["b"]}')
for vis, lab in (('v1', 'V1 AUC (observed dose), in range'), ('v2', 'V2 AUC, new dose 1-8 mg, in range')):
    print(f'== {lab}')
    C = {k: {s: net_cell(path[k](s), vis) for s in seeds[k]} for k in path}
    for k in path:
        print(f'   {name[k]:32s} ' + fmt(agg(list(C[k].values())), show_cov=(vis == 'v2')))
    fm = np.array([C['film'][s]['med'] for s in seeds['film']])
    am = np.array([C['a15'][s]['med'] for s in seeds['a15']])
    if len(am) == 0 or len(fm) == 0:
        continue
    print(f'   seed level: A medians {np.round(am, 2).tolist()} vs FiLM {np.round(fm, 2).tolist()}; mean A - FiLM '
          f'{am.mean() - fm.mean():+.2f} pt; A seeds below every FiLM seed {int(np.sum(am < fm.min()))}/{len(am)}, above every '
          f'FiLM seed {int(np.sum(am > fm.max()))}/{len(am)}'
          + (f'; Mann-Whitney on seed medians p = {mannwhitneyu(am, fm, alternative="two-sided").pvalue:.2g}'
             if len(am) >= 2 and len(fm) >= 2 else ''))
    ref = json.load(open(path['film'](seeds['film'][0])))['per_patient'][f'true_auc_{vis}']
    if not all(np.allclose(ref, json.load(open(path[k](s)))['per_patient'][f'true_auc_{vis}'])
               for k in ('film', 'a15') for s in seeds[k]):
        print('   paired comparisons skipped: patient order differs'); continue
    diffs, below, above = [], 0, 0
    for sa in seeds['a15']:
        for sf in seeds['film']:
            da, df = C['a15'][sa]['abs'], C['film'][sf]['abs']
            lo, hi = boot_ci(da, df)
            diffs.append(100 * (np.median(da) - np.median(df)))
            below += hi < 0; above += lo > 0
    print(f'   every A seed vs every FiLM seed, same patients ({len(diffs)} pairs): median diff {np.median(diffs):+.2f} pt '
          f'(range {min(diffs):+.2f} to {max(diffs):+.2f}); 95% bootstrap CI entirely below 0 in {below}, above 0 in {above}')
    a = agg(list(C['a15'].values()))['abs_mean_over_seeds']; f = agg(list(C['film'].values()))['abs_mean_over_seeds']
    lo, hi = boot_ci(a, f)
    print(f'   seed-averaged |err| per patient: median diff {100 * (np.median(a) - np.median(f)):+.2f} pt '
          f'[95% boot {lo:+.2f}, {hi:+.2f}], A closer for {100 * np.mean(a < f):.1f}%, Wilcoxon p = {wilcoxon(a, f).pvalue:.2g} '
          f'(patient-level: ignores seed variance)')
