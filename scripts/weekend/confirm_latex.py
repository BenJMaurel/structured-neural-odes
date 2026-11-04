#!/usr/bin/env python3
"""LaTeX rows for main-text Table 3 and Supplementary Table S6 from the confirmation runs, so that no
number is copied by hand. Same runs, exclusion (diverged training, |training-regimen error| > 100 %, left out with its pair) and
metrics as confirm_table.py. Prints the rows and writes results/regimen_dh/confirm/confirm_latex.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, json
import numpy as np
os.chdir(REPO)
RD, RT = 'results/regimen_dh/confirm', f'{REPO}/third_party/baaz_ebvae/ours/regimen/res_titr/confirm'
ORIG = ('amt350', 'amt600', 'amt850', 'time400', 'time800'); HELD = ('c_amt500', 'c_amt1000', 'c_load', 'c_missed', 'c_extra')
CONDS = [('sparse randomised', 'ctrl', 'sparse', range(5, 15)), ('sparse titrated', 'titr0.8', 'sparse', range(5, 15)), ('dense randomised', 'ctrl', 'dense', range(3, 8))]


def cell(p):
    r = json.load(open(p))['individual']; e = {k: np.array(r[k]['auc_err_per_patient']) for k in ('amt350', 'amt850')}
    return dict(u=np.mean([r[k]['auc_abs_median'] for k in ORIG]), h=np.mean([r[k]['auc_abs_median'] for k in HELD]), t=r['train']['auc_abs_median'],
                q=float(np.median((1 + e['amt850']) / (1 + e['amt350']))), load=r['c_load']['auc_abs_median'])


pm = lambda v: f'${np.mean(v):.1f}\\pm{np.std(v, ddof=1):.1f}$'
R = {}
for name, sub, d, seeds in CONDS:
    A, B, left = [], [], []
    for s in seeds:
        a, b = cell(f'{RD}/{sub}/Bfixkp20n/N100_{d}_s{s}.json'), cell(f'{RT}/{sub}/N100_{d}_s{s}.json')
        if a['t'] > 100 or b['t'] > 100:
            left.append(s); continue
        A.append(a); B.append(b)
    R[name] = (A, B, left)
L = ['% Table 3 cells (Unseen, Held-out per design): EB-VAE row, then dose events row']
for who, idx in (('EB-VAE', 1), ('Dose events (ours)', 0)):
    cells = []
    for name, *_ in CONDS:
        X = R[name][idx]
        u, h = pm([x['u'] for x in X]), pm([x['h'] for x in X])
        if idx == 0:
            u, h = u.replace('$', '$\\mathbf{', 1)[:-1] + '}$', h.replace('$', '$\\mathbf{', 1)[:-1] + '}$'
        cells += [u, h]
    L.append(f'{who} & ' + ' & '.join(cells) + ' \\\\')
L.append('')
L.append('% Table S6 rows: design & model & unseen & held-out & training & loading dose & dose response & lower (unseen, held-out)')
for name, *_ in CONDS:
    A, B, left = R[name]
    lo_u = sum(a['u'] < b['u'] for a, b in zip(A, B)); lo_h = sum(a['h'] < b['h'] for a, b in zip(A, B))
    for who, X in (('Dose events', A), ('EB-VAE', B)):
        L.append(f'{name} ({len(X)}) & {who} & {pm([x["u"] for x in X])} & {pm([x["h"] for x in X])} & {pm([x["t"] for x in X])} & '
                 f'{pm([x["load"] for x in X])} & ${np.mean([x["q"] for x in X]):.2f}$ & '
                 + (f'{lo_u}/{len(A)}, {lo_h}/{len(A)}' if who == 'Dose events' else '') + ' \\\\')
    L.append(f'%   left out: {left}; loading-dose range ours {min(a["load"] for a in A):.1f}-{max(a["load"] for a in A):.1f}, '
             f'EB-VAE {min(b["load"] for b in B):.1f}-{max(b["load"] for b in B):.1f}')
open(f'{RD}/confirm_latex.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
