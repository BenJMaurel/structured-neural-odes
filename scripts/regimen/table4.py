#!/usr/bin/env python3
"""Table 4 of ndm.tex: on the regimen benchmark of Baaz et al., linear simulator, the
EB-VAE (dose in its vector field) vs the dose-event model of the main text (free drug network, jump b(p), noise 20 mg/L,
dose-normalised encoder = Bfixkp20n) vs the dose-conditioned field with the same encoder (dosecond_n: dose as an input of
the vector field through a decaying trace, no jump; dh_regimen.py --dose-cond --enc-dose norm --obsrv-mgl 20).
  unseen   = mean over the 5 unseen regimens of the median |AUC(0-30 h) error| (%)
  observed = the same on the training regimen;  mean +- sd over seeds (ddof 1);  unseen > 1000 % = diverged, left out
Paired differences against the EB-VAE on common seeds (no p-values. No dose elasticity.
Writes results/regimen_dh/table4.txt."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
R = os.path.join(ROOT, 'results/regimen_dh')
EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'
UNSEEN = ('amt350', 'amt600', 'amt850', 'time400', 'time800')
SEEDS = {'sparse': range(5), 'titr': range(3), 'dense': range(3)}
MODELS = [('ebvae', 'EB-VAE (dose in the vector field)'), ('Bfixkp20n', 'dose-event model (main text)'),
          ('dcn', 'dose-conditioned field (same encoder)')]


def path(model, col, s):
    d = 'dense' if col == 'dense' else 'sparse'
    tag = f'N100_{d}_s{s}.json'
    if model == 'ebvae':
        p = f'{EB}/res_titr/{"titr0.8" if col == "titr" else "ctrl"}/{tag}'
        return p if os.path.exists(p) or col == 'titr' else f'{EB}/res/N100_{d}' + (f'_s{s}' if s else '') + '.json'
    if model == 'dcn':
        return f'{R}/dosecond_n/{"titr0.8/" if col == "titr" else ""}{tag}'
    return f'{R}/{"titr0.8/" if col == "titr" else ""}{model}/{tag}'


def per_seed(model, col):
    out = {}
    for s in SEEDS[col]:
        p = path(model, col, s)
        if not os.path.exists(p):
            continue
        ind = json.load(open(p))['individual']
        u = float(np.mean([ind[g]['auc_abs_median'] for g in UNSEEN]))
        out[s] = dict(u=u, o=float(ind['train']['auc_abs_median']), div=not np.isfinite(u) or u > 1000)
    return out


def ms(v):
    return '-' if not v else (f'{np.mean(v):.1f} +- {np.std(v, ddof=1):.1f}' if len(v) > 1 else f'{v[0]:.1f}')


lines = [__doc__.strip().split('\n')[0], '']
P = {(m, c): per_seed(m, c) for m, _ in MODELS for c in SEEDS}
for col in ('sparse', 'titr', 'dense'):
    lines.append(f'=== linear simulator, {col} ({"titrated arms, sparse sampling" if col == "titr" else "randomised arms"})')
    for m, lab in MODELS:
        r = P[(m, col)]; ok = {s: v for s, v in r.items() if not v['div']}
        u, o = [v['u'] for v in ok.values()], [v['o'] for v in ok.values()]
        div = sorted(s for s, v in r.items() if v['div'])
        lines.append(f'   {lab:40s} n {len(ok)}  unseen {ms(u):>12s}  training regimen {ms(o):>12s}  gap {ms([a - b for a, b in zip(u, o)]):>12s}'
                     f'   per seed {" ".join(f"s{s} {v[chr(117)]:.1f}" for s, v in sorted(ok.items()))}' + (f'  DIVERGED {div}' if div else ''))
    for m, lab in MODELS[1:]:
        e, x = P[('ebvae', col)], P[(m, col)]
        common = sorted(s for s in e if s in x and not e[s]['div'] and not x[s]['div'])
        d = [x[s]['u'] - e[s]['u'] for s in common]
        if d:
            lines.append(f'      paired {lab} - EB-VAE: {np.mean(d):+.1f} pt, lower in {sum(v < 0 for v in d)}/{len(d)} seeds')
    lines.append('')
for m, lab in MODELS:   # titration: change of the unseen error and of the unseen - training gap, same seeds 0-2
    r, t = P[(m, 'sparse')], P[(m, 'titr')]
    common = sorted(s for s in t if s in r and not r[s]['div'] and not t[s]['div'])
    du = [t[s]['u'] - r[s]['u'] for s in common]
    dg = [(t[s]['u'] - t[s]['o']) - (r[s]['u'] - r[s]['o']) for s in common]
    lines.append(f'titration effect, {lab:40s} seeds {common}: unseen {ms([r[s]["u"] for s in common])} -> {ms([t[s]["u"] for s in common])}'
                 f' (change {ms(du)}); gap unseen - training change {ms(dg)}')
open(f'{R}/table4.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
