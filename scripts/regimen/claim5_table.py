#!/usr/bin/env python3
"""Claim 5 with seeds: dose-event models vs the EB-VAE on Baaz's regimen test, N100 randomised arms, linear
and saturable (Km 500) simulators, sparse seeds 0-4 / dense seeds 0-2 (seed 0's randomised arms correlate +0.27 with log ke).
Per model: unseen-regimen error (mean of the median |AUC 0-30 h err| over the 5 unseen regimens), observed-regimen error, dose
elasticity 350 -> 850 mg [truth], arm-switch signed errors (observed-800 -> 350 / observed-400 -> 850; 0 / 0 = right dose
scaling); mean +- sd over seeds and the per-seed values. A run with unseen error > 1000 % is reported as DIVERGED and left out of
the means. Paired vs the EB-VAE on the same seeds: mean difference, number of seeds where the model is better, Wilcoxon p.

    claim5_table.py > results/regimen_dh/claim5_table.txt
"""
import io, sys, contextlib
import numpy as np
from scipy import stats
sys.path.insert(0, 'scripts/regimen')
with contextlib.redirect_stdout(io.StringIO()):
    import bfix_table as BT          # patches v3_table.path for the Bfix* models
V = BT.V
MODELS = [('mapbe', 'MAP-BE, true model (ceiling)'), ('ebvae', 'EB-VAE (dose in the field)'),
          ('lam1000', 'improved B (linear + penalised corrections)'), ('Bfixkp20', 'free-flow B + jump a*b(p), 20 mg/L')]
SEEDS = {'sparse': (0, 1, 2, 3, 4), 'dense': (0, 1, 2)}


def per_seed(model, sim, d):
    rows = {}
    for s in SEEDS[d]:
        c = V.cell(model, sim, 'ctrl', d, (s,))
        if c is None:
            continue
        a3, a8 = BT.arm_split(model, sim, d, s)
        rows[s] = dict(unseen=c['unseen'], fact=c['fact'], el=c['el'], el_true=c['el_true'], a3=a3, a8=a8)
    return rows


def ms(v, f='{:.1f}'):
    v = [x for x in v if np.isfinite(x)]
    return (f.format(np.mean(v)) + ' +- ' + f.format(np.std(v, ddof=1) if len(v) > 1 else 0)) if v else '-'


def main():
    print(__doc__.strip().split('\n\n')[0])
    for sim, sname in (('lin', 'LINEAR simulator (true elasticity ~0.97)'), ('mm500', 'SATURABLE simulator, Km 500 (true elasticity ~1.26)')):
        for d in ('sparse', 'dense'):
            print(f'\n=== {sname}, N100 {d}, seeds {SEEDS[d]}')
            print(f'   {"model":44s} {"n":>2s} {"unseen":>14s} {"observed":>14s} {"elasticity":>14s} {"800->350":>9s} {"400->850":>9s}   per-seed unseen')
            R = {m: per_seed(m, sim, d) for m, _ in MODELS}
            for m, name in MODELS:
                r = R[m]
                ok = {s: v for s, v in r.items() if v['unseen'] < 1000}
                bad = sorted(set(r) - set(ok))
                if not r:
                    print(f'   {name:44s}  - (no run)'); continue
                el_true = np.mean([v['el_true'] for v in r.values()])
                print(f'   {name:44s} {len(ok):2d} {ms([v["unseen"] for v in ok.values()]):>14s} {ms([v["fact"] for v in ok.values()]):>14s} '
                      f'{ms([v["el"] for v in ok.values()], "{:.2f}"):>9s} [{el_true:.2f}] {ms([v["a3"] for v in ok.values()], "{:+.0f}"):>9s} '
                      f'{ms([v["a8"] for v in ok.values()], "{:+.0f}"):>9s}   ' + ' '.join(f's{s} {v["unseen"]:.1f}' for s, v in sorted(ok.items()))
                      + (f'   DIVERGED: s{bad}' if bad else ''))
            e = {s: v['unseen'] for s, v in R['ebvae'].items() if v['unseen'] < 1000}
            for m, name in MODELS[2:]:
                S = sorted(set(e) & {s for s, v in R[m].items() if v['unseen'] < 1000})
                if len(S) < 2:
                    continue
                dd = np.array([R[m][s]['unseen'] - e[s] for s in S])
                p = stats.wilcoxon(dd).pvalue if len(S) >= 3 and np.any(dd != 0) else np.nan
                print(f'   paired {name:37s} - EB-VAE: {np.mean(dd):+6.1f} pt (better in {int((dd < 0).sum())}/{len(S)} seeds, Wilcoxon p {p:.2g})')


if __name__ == '__main__':
    main()
