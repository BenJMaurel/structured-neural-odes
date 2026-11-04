#!/usr/bin/env python3
"""Tables for the overnight queue 03->04/10 (queue_night.sh). Uses whatever seeds exist.

    night_tables.py tslow   FiLM vs variant A (--orth-split 1,5) on the 2nd windowed cohort (confound_vc00_s4_tslow_win,
                            Student-t + 30 % slow absorbers; N=100 subset made 03/10 22:10), 1080 test patients
    night_tables.py sweep   variant A split sweep on windowed pnoise: --orth-split 1,K for K = 3..7 (physiology 6..2 dims)
                            vs FiLM; ep6000, 1100 test patients
In range first (V1, V2 1-8 mg: question), then out of range. Seed level first (carries seed variance), then
paired per patient (same patients, checked on the true AUC arrays; ignores seed variance)."""
import os, sys, json, numpy as np
mode = sys.argv[1] if len(sys.argv) > 1 else 'sweep'
sys.argv = sys.argv[:1]; sys.path.insert(0, 'scripts/ndm')
from ndm_tables import net_cell, agg, fmt
from scipy.stats import wilcoxon, mannwhitneyu
rng = np.random.default_rng(0)
REGS = (('in', 'v1', 'V1 AUC (observed dose), in range'), ('in', 'v2', 'V2 AUC, new dose 1-8 mg, in range'),
        ('hi', 'v2', 'V2 AUC, above range 10-12 mg'), ('lo', 'v2', 'V2 AUC, below range 0.25-0.5 mg'))
if mode == 'tslow':
    E = 'results/s4_tslow_win/eval'
    nets = {'film': ('FiLM, unconstrained', lambda s, r: f'{E}/ep006000_film_n100_s{s}_on_{r}.json'),
            'a15': ('A: FiLM + orth-split 1/5/4', lambda s, r: f'{E}/ep006000_osplit1-5_n100_s{s}_on_{r}.json')}
    test = ['a15']
elif mode == 'sweep':
    E3, ES, EA = 'results/s4_pnoise_win/eval', 'results/s4_seeds/eval', 'results/s4_orth/eval'
    nets = {'film': ('FiLM, unconstrained (Table 3)', lambda s, r: f'{E3 if s <= 3 else ES}/ep006000_film_n100_s{s}_on_{r}.json')}
    for K in (3, 4, 5, 6, 7):
        nets[f'a1{K}'] = (f'A 1/{K}/{10 - 1 - K} (FiLM/dyn/phys)',
                          (lambda K: lambda s, r: f'{EA}/ep006000_osplit1-{K}_n100_s{s}_on_{r}.json')(K))
    test = [f'a1{K}' for K in (3, 4, 5, 6, 7)]
else:
    sys.exit(f'unknown mode {mode}')


def boot_ci(da, db, n_boot=1000):
    n = len(da)
    b = [100 * (np.median(da[i]) - np.median(db[i])) for i in (rng.integers(0, n, n) for _ in range(n_boot))]
    return np.percentile(b, [2.5, 97.5])


for reg, vis, lab in REGS:
    print(f'== {lab}')
    C, S = {}, {}
    for k, (nm, p) in nets.items():
        S[k] = [s for s in range(1, 11) if os.path.exists(p(s, reg))]
        C[k] = {s: net_cell(p(s, reg), vis) for s in S[k]}
        print(f'   {nm:34s} ' + fmt(agg(list(C[k].values())), show_cov=(vis == 'v2')))
    if not S['film']:
        continue
    fm = np.array([C['film'][s]['med'] for s in S['film']])
    ref = json.load(open(nets['film'][1](S['film'][0], reg)))['per_patient'][f'true_auc_{vis}']
    for k in test:
        if not S[k]:
            continue
        am = np.array([C[k][s]['med'] for s in S[k]])
        line = (f'   {k}: seed medians {np.round(am, 2).tolist()} vs FiLM mean {fm.mean():.2f}; mean diff {am.mean() - fm.mean():+.2f} pt; '
                f'below every FiLM seed {int(np.sum(am < fm.min()))}/{len(am)}, above every {int(np.sum(am > fm.max()))}/{len(am)}')
        if len(am) >= 2 and len(fm) >= 2:
            line += f'; Mann-Whitney p = {mannwhitneyu(am, fm, alternative="two-sided").pvalue:.2g}'
        print(line)
        if reg == 'in' and all(np.allclose(ref, json.load(open(nets[k][1](s, reg)))['per_patient'][f'true_auc_{vis}']) for s in S[k]):
            a, f = agg(list(C[k].values()))['abs_mean_over_seeds'], agg(list(C['film'].values()))['abs_mean_over_seeds']
            lo, hi = boot_ci(a, f)
            print(f'      paired, seed-averaged |err|: median diff {100 * (np.median(a) - np.median(f)):+.2f} pt [95% boot {lo:+.2f}, '
                  f'{hi:+.2f}], closer for {100 * np.mean(a < f):.1f}%, Wilcoxon p = {wilcoxon(a, f).pvalue:.2g}')
