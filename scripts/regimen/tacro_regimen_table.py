#!/usr/bin/env python3
"""Tacrolimus transposition of the regimen benchmark: the dose-event model of Table 4 (free drug flow, jump
a*b(p), fixed noise 4 ng/mL, dose-normalised encoder) vs the EB-VAE (their pipeline, covariates CYP3A5 + Ht, run_regimen_tacro.py)
on gen_tacro_regimens.py (scenario-4 simulator, Prograf q12h, arms 2/4 mg); same patients for both models at each seed.
Same definitions as table4.py:
  unseen   = mean over the 5 unseen regimens (1.75 / 3 / 4.25 mg q12h; 2 or 4 mg at 0, 4.5, 12 h) of the median |AUC(0-45 h) error| (%)
  training = the same on the training regimen;  mean +- sd over seeds (ddof 1);  unseen > 1000 % = diverged, left out
plus the per-regimen median |error| and median signed error (bias), paired differences vs the EB-VAE on common seeds, and the
titration effect on common seeds.
    python3 tacro_regimen_table.py [root] [--scen 2]   -> <root>/table.txt   (default results/regimen_dh/tacro; --scen 2 =
    scenario 2, linear elimination: results/regimen_dh/tacro_s2, res_tacro_s2, dose-event noise 3.1 ng/mL)"""
import os, sys, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ARGS = [a for a in sys.argv[1:]]
SCEN = 4
if '--scen' in ARGS:
    k = ARGS.index('--scen'); SCEN = int(ARGS[k + 1]); del ARGS[k:k + 2]
os.environ['TACRO_SCENARIO'] = str(SCEN)
import gen_tacro_regimens as G
R = ARGS[0] if ARGS else os.path.join(ROOT, 'results/regimen_dh/tacro' + ('_s2' if SCEN == 2 else ''))
SEEDS = {'sparse': range(5), 'visit': range(5), 'dense': range(3), 'titr': range(3)}
EB = os.path.expanduser('third_party/baaz_ebvae/ours/regimen/res_tacro' + ('_s2' if SCEN == 2 else ''))
MAIN = 'Bfixkp3.1n' if SCEN == 2 else 'Bfixkp4n'
MODELS = [('ebvae', 'EB-VAE (dose in the vector field)'), (MAIN, 'dose-event model (Table 4 architecture)')]


def path(model, col, s):
    tag = f'N100_{"sparse" if col == "titr" else col}_s{s}.json'
    if model == 'ebvae':
        return f'{EB}/{"titr0.8" if col == "titr" else "ctrl"}/{tag}'
    return f'{R}/{"titr0.8/" if col == "titr" else ""}{model}/{tag}'


def per_seed(model, col):
    out = {}
    for s in SEEDS[col]:
        p = path(model, col, s)
        if not os.path.exists(p):
            continue
        d = json.load(open(p)); ind = d['individual']
        u = float(np.mean([ind[g]['auc_abs_median'] for g in G.UNSEEN]))
        out[s] = dict(u=u, o=float(ind['train']['auc_abs_median']), div=not np.isfinite(u) or u > 1000,
                      reg={g: (ind[g]['auc_abs_median'], ind[g]['auc_bias_median']) for g in ind},
                      corr=d.get('titrate_corr'))
    return out


def ms(v):
    return '-' if not v else (f'{np.mean(v):.1f} +- {np.std(v, ddof=1):.1f}' if len(v) > 1 else f'{v[0]:.1f}')


lines = [__doc__.strip().split('\n')[0], f'SCENARIO {SCEN} ({"linear elimination" if SCEN == 2 else "Michaelis-Menten elimination"})', '']
P = {(m, c): per_seed(m, c) for m, _ in MODELS for c in SEEDS}
for col, lab in (('sparse', 'sparse, randomised arms'), ('visit', 'intensive PK visit (12 samples), randomised arms'),
                 ('dense', 'dense, randomised arms'), ('titr', 'sparse, titrated arms (rho 0.8)')):
    lines.append(f'=== {lab}')
    for m, mlab in MODELS:
        r = P[(m, col)]; ok = {s: v for s, v in r.items() if not v['div']}
        u, o = [v['u'] for v in ok.values()], [v['o'] for v in ok.values()]
        div = sorted(s for s, v in r.items() if v['div'])
        lines.append(f'  {mlab}: n {len(ok)}/{len(SEEDS[col])}  unseen {ms(u)}  training regimen {ms(o)}  gap {ms([a - b for a, b in zip(u, o)])}'
                     + (f'  DIVERGED {div}' if div else '')
                     + (f'  corr(log Vmax, 4 mg) {ms([v["corr"] for v in ok.values()])}' if col == 'titr' and ok and m != 'ebvae' else ''))
        if ok:
            lines.append('    per seed unseen: ' + '  '.join(f's{s} {v["u"]:.1f}' for s, v in sorted(ok.items())))
            for g in ('train',) + G.UNSEEN:
                a = [v['reg'][g][0] for v in ok.values()]; b = [v['reg'][g][1] for v in ok.values()]
                lines.append(f'    {g:9s} median |err| {ms(a):>12s}   median bias {ms(b):>12s}')
    e, x = P[('ebvae', col)], P[(MAIN, col)]
    common = sorted(s for s in e if s in x and not e[s]['div'] and not x[s]['div'])
    d = [x[s]['u'] - e[s]['u'] for s in common]
    if d:
        lines.append(f'  paired dose-event - EB-VAE (unseen), seeds {common}: {np.mean(d):+.1f} pt, lower in {sum(v < 0 for v in d)}/{len(d)} seeds')
    lines.append('')
for m, mlab in MODELS:   # titration: same seeds, same patients, only the arm assignment changes
    r, t = P[(m, 'sparse')], P[(m, 'titr')]
    common = sorted(s for s in t if s in r and not r[s]['div'] and not t[s]['div'])
    if common:
        lines.append(f'titration effect, {mlab}, seeds {common}: unseen {ms([r[s]["u"] for s in common])} -> {ms([t[s]["u"] for s in common])}; '
                     f'training {ms([r[s]["o"] for s in common])} -> {ms([t[s]["o"] for s in common])}; '
                     f'gap change {ms([(t[s]["u"] - t[s]["o"]) - (r[s]["u"] - r[s]["o"]) for s in common])}')
os.makedirs(R, exist_ok=True)
open(f'{R}/table.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
