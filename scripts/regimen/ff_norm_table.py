#!/usr/bin/env python3
"""Free-flow dose-event model with a DOSE-NORMALISED encoder: does removing the dose from the encoder fix the
free flow's under-response to dose? Sparse seeds 0-4, randomised arms, linear and saturable (Km 500) simulators, N100.
Rows: EB-VAE, improved B (lam 1000), free-flow B + jump a*b(p) 20 mg/L with the raw encoder (Bfixkp20, claim-5 runs) and with
--enc-dose norm (Bfixkp20n). Unseen = mean of the median |AUC 0-30 h err| over the 5 unseen regimens; observed = training
regimen; elasticity 350 -> 850 mg [truth]; arm switch = signed median err observed-800 -> 350 / observed-400 -> 850 (0 / 0 =
right dose scaling, + / - = the exposure level stays with the patient). Runs with unseen error > 1000 % = DIVERGED, left out.
Paired: same seeds, mean difference of the unseen error, seeds where the row is better.

    ff_norm_table.py > results/regimen_dh/ff_norm_table.txt
"""
import io, sys, contextlib
import numpy as np
sys.path.insert(0, 'scripts/regimen')
with contextlib.redirect_stdout(io.StringIO()):
    import bfix_table as BT
V = BT.V
_path = BT.path


def path(model, sim, cond, d, s):
    if model == 'Bfixkp20n':
        return f'{V.R}/{"mm500/" if sim != "lin" else ""}{model}/N100_{d}_s{s}.json'
    return _path(model, sim, cond, d, s)


V.path = path; BT.path = path
MODELS = [('ebvae', 'EB-VAE (dose in the field)'), ('lam1000', 'improved B (linear + penalised corrections)'),
          ('Bfixkp20', 'free-flow B, raw encoder'), ('Bfixkp20n', 'free-flow B, dose-normalised encoder')]
SEEDS = (0, 1, 2, 3, 4)


def main():
    print(__doc__.strip().split('\n\n')[0])
    for sim, sname in (('lin', 'LINEAR simulator'), ('mm500', 'SATURABLE simulator, Km 500')):
        print(f'\n=== {sname}, N100 sparse, seeds {SEEDS}')
        print(f'   {"model":44s} {"n":>2s} {"unseen":>13s} {"observed":>13s} {"elasticity":>16s} {"800->350":>8s} {"400->850":>8s}   per seed (unseen / elasticity)')
        R = {}
        for m, name in MODELS:
            r = {}
            for s in SEEDS:
                c = V.cell(m, sim, 'ctrl', 'sparse', (s,))
                if c is not None:
                    a3, a8 = BT.arm_split(m, sim, 'sparse', s)
                    r[s] = dict(unseen=c['unseen'], fact=c['fact'], el=c['el'], el_true=c['el_true'], a3=a3, a8=a8)
            R[m] = r
            ok = {s: v for s, v in r.items() if v['unseen'] < 1000}; bad = sorted(set(r) - set(ok))
            if not ok:
                print(f'   {name:44s}  - (no run)' + (f'   DIVERGED: s{bad}' if bad else '')); continue
            f = lambda k, fmt='{:.1f}': (fmt.format(np.mean([v[k] for v in ok.values()])) + ' +- ' +
                                       fmt.format(np.std([v[k] for v in ok.values()], ddof=1) if len(ok) > 1 else 0))
            et = np.mean([v['el_true'] for v in ok.values()])
            print(f'   {name:44s} {len(ok):2d} {f("unseen"):>13s} {f("fact"):>13s} {f("el", "{:.2f}"):>11s} [{et:.2f}] '
                  f'{np.mean([v["a3"] for v in ok.values()]):+8.0f} {np.mean([v["a8"] for v in ok.values()]):+8.0f}   '
                  + ' '.join(f's{s} {v["unseen"]:.1f}/{v["el"]:.2f}' for s, v in sorted(ok.items())) + (f'   DIVERGED: s{bad}' if bad else ''))
        new = {s: v['unseen'] for s, v in R['Bfixkp20n'].items() if v['unseen'] < 1000}
        for ref, lab in (('Bfixkp20', 'raw encoder'), ('ebvae', 'EB-VAE'), ('lam1000', 'improved B')):
            S = sorted(set(new) & {s for s, v in R[ref].items() if v['unseen'] < 1000})
            if S:
                dd = np.array([new[s] - R[ref][s]['unseen'] for s in S])
                print(f'   paired dose-normalised free-flow B - {lab:12s}: {dd.mean():+6.1f} pt (better in {int((dd < 0).sum())}/{len(S)} seeds)')


if __name__ == '__main__':
    main()
