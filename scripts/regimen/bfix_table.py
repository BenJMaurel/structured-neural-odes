#!/usr/bin/env python3
"""Free-flow B with FIXED noise.
Original B (dh_regimen.py: anchored free flow in the drug subspace, kick a*b, residual decoder) with ONE change, the noise
sd fixed at B2's 0.05 x max training observation (--obsrv-std 0.05) instead of learned. Baaz's regimen test, N100, sparse
seed 1 / dense seed 0 (the cells of v3_table.py), linear and saturable (Km 500) simulators, randomised and titrated (rho
0.8) training arms. Columns: noise sd (mg/L), observed-regimen error, mean unseen-regimen error (median |AUC 0-30 h err| %,
new amounts / new times), dose elasticity 350 -> 850 mg [truth], median signed error observed-on-800 -> 350 mg /
observed-on-400 -> 850 mg (patient-route signature: + / -; right dose scaling 0 / 0), then the titrated run's unseen error
and elasticity.

    bfix_table.py > results/regimen_dh/bfix_table.txt
"""
import os, sys, io, json, contextlib
import numpy as np
sys.path.insert(0, 'scripts/regimen')
with contextlib.redirect_stdout(io.StringIO()):   # v3_table prints its own table at import
    import v3_table as V

_path = V.path
def path(model, sim, cond, d, s):
    if model in ('Bfix', 'Bfixkp', 'Bfix20', 'Bfixkp20'):
        return f'{V.R}/{"mm500/" if sim != "lin" else ""}{"titr0.8/" if cond == "titr" else ""}{model}/N100_{d}_s{s}.json'
    return _path(model, sim, cond, d, s)
V.path = path

MODELS = [('mapbe', 'MAP-BE, true model (ceiling)'), ('ebvae', 'EB-VAE'), ('B', 'B original (free flow, LEARNED noise)'),
          ('Bfix', 'B free flow, FIXED noise  <- new'), ('Bfixkp', 'B free flow, fixed noise, jump a*b(p) <- new'), ('Bfix20', 'B free flow, noise 20 mg/L'), ('Bfixkp20', 'B free flow, noise 20 mg/L, jump a*b(p)'), ('B2', 'B2 (linear, fixed noise)'),
          ('lam0', 'improved B, penalty 0'), ('lam1000', 'improved B, penalty 1000')]


def arm_split(model, sim, d, s):
    j = V.load(path(model, sim, 'ctrl', d, s))
    if j is None:
        return np.nan, np.nan
    r = j['individual']
    e350, e850 = np.array(r['amt350']['auc_err_per_patient']), np.array(r['amt850']['auc_err_per_patient'])
    return 100 * np.median(e350[1::2]), 100 * np.median(e850[0::2])   # test patients alternate 400 (even) / 800 (odd)


def noise(model, sim, d, s):
    j = V.load(path(model, sim, 'ctrl', d, s))
    return next((j[k] for k in ('noise_sd', 'noise_sd_final', 'obsrv_std_mgL') if j and k in j), np.nan)


def main():
    print(__doc__.strip().split('\n\n')[0].replace('\n', ' '))
    for sim, sname in (('lin', 'LINEAR simulator'), ('mm500', 'SATURABLE simulator (Km 500)')):
        for d, seeds in V.CONFIGS:
            s = seeds[0]
            print(f'\n=== {sname}, N100 {d} s{s}')
            print(f'   {"model":40s} {"noise":>6s} | {"obs.":>5s} {"unseen":>7s} ({"amts":>5s}/{"times":>5s}) {"elast.":>13s} | '
                  f'{"800->350":>8s} {"400->850":>8s} | {"titrated: unseen":>16s} {"elast.":>6s}')
            for mk, name in MODELS:
                c = V.cell(mk, sim, 'ctrl', d, seeds)
                t = V.cell(mk, sim, 'titr', d, seeds)
                if c is None and t is None:
                    continue
                a3, a8 = arm_split(mk, sim, d, s)
                nz = noise(mk, sim, d, s)
                base = (f'{c["fact"]:5.1f} {c["unseen"]:7.1f} ({c["amts"]:5.1f}/{c["times"]:5.1f}) {c["el"]:6.2f} [{c["el_true"]:.2f}]'
                        if c else f'{"-":>5s} {"-":>7s} ({"-":>5s}/{"-":>5s}) {"-":>13s}')
                tt = f'{t["unseen"]:16.1f} {t["el"]:6.2f}' if t else f'{"-":>16s} {"-":>6s}'
                print(f'   {name:40s} {nz:6.1f} | {base} | {a3:+8.1f} {a8:+8.1f} | {tt}')


if __name__ == '__main__':
    main()
