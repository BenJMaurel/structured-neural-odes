#!/usr/bin/env python3
"""OPTION (b) on the regimen benchmark of Baaz et al. (linear simulator; the three cells of main-text Table 3; ):
does reading the dose value through zero-initialised weights (--enc-dose normd, Bfixkp20nd) bring back the arm leak of the raw
encoder? Rows: EB-VAE, dose-event model with the raw encoder (Bfixkp20; no titrated runs), with the dose-normalised encoder
(Bfixkp20n = Table 3) and with option (b) (Bfixkp20nd). Same definitions as baaz_arm_split.py (standalone copy, so nothing else is
rewritten): unseen = mean of the median |AUC 0-30 h err| over the 5 unseen regimens; training = the training regimen; model / true =
median over test patients of the model's AUC(850)/AUC(350) for the same patient divided by the true one (1 = right); cross-arm =
median signed error observed on 800 -> predicted at 350 | observed on 400 -> predicted at 850 (0 | 0 = right dose scaling).
Paired: same seed, difference of the unseen error and of model / true.
Writes results/regimen_dh/encdoseval_table.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
R = os.path.join(ROOT, 'results/regimen_dh'); EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'
SEEDS = {'sparse': range(5), 'titr': range(3), 'dense': range(3)}
UNSEEN = ('amt350', 'amt600', 'amt850', 'time400', 'time800')
MODELS = (('ebvae', 'EB-VAE'), ('Bfixkp20', 'ours, raw encoder'), ('Bfixkp20n', 'ours, dose-normalised (T3)'),
          ('Bfixkp20nd', 'ours, option (b)'))


def path(model, col, s):          # as baaz_arm_split.py / table4.py
    d = 'dense' if col == 'dense' else 'sparse'; tag = f'N100_{d}_s{s}.json'
    if model == 'ebvae':
        p = f'{EB}/res_titr/{"titr0.8" if col == "titr" else "ctrl"}/{tag}'
        return p if os.path.exists(p) or col == 'titr' else f'{EB}/res/N100_{d}' + (f'_s{s}' if s else '') + '.json'
    return f'{R}/{"titr0.8/" if col == "titr" else ""}{model}/{tag}'


def truth(col, s):
    r = json.load(open(path('ebvae', col, s)))['individual']
    return {g: np.array(r[g]['auc_pred_per_patient']) / (1 + np.array(r[g]['auc_err_per_patient'])) for g in ('amt350', 'amt850')}


def cell(m, col, s):
    p = path(m, col, s)
    if not os.path.exists(p):
        return None
    r = json.load(open(p))['individual']
    u = float(np.mean([r[g]['auc_abs_median'] for g in UNSEEN]))
    if not np.isfinite(u) or u > 1000:
        return dict(diverged=True)
    e = {g: np.array(r[g]['auc_err_per_patient']) for g in ('amt350', 'amt850')}
    q = (1 + e['amt850']) / (1 + e['amt350'])
    return dict(diverged=False, unseen=u, train=float(r['train']['auc_abs_median']), rel=float(np.median(q)),
                a3=100 * float(np.median(e['amt350'][1::2])), a8=100 * float(np.median(e['amt850'][0::2])))


def msd(v, fmt='{:.1f}'):
    return fmt.format(np.mean(v)) + ' +- ' + fmt.format(np.std(v, ddof=1) if len(v) > 1 else 0.0)


lines = [__doc__.strip().split('\n')[0], '']
for col, lab in (('sparse', 'sparse, randomised arms'), ('titr', 'sparse, TITRATED arms'), ('dense', 'dense, randomised arms')):
    lines.append(f'=== linear simulator, {lab}, seeds {list(SEEDS[col])}')
    lines.append(f'   {"model":28s} {"n":>2s} {"unseen":>13s} {"training":>13s} {"model / true":>18s} {"800->350 | 400->850":>20s}   per seed unseen (rel)')
    C = {}
    for m, name in MODELS:
        c = {s: cell(m, col, s) for s in SEEDS[col]}
        c = {s: v for s, v in c.items() if v is not None}
        bad = sorted(s for s, v in c.items() if v['diverged']); ok = {s: v for s, v in c.items() if not v['diverged']}
        C[m] = ok
        if not ok:
            lines.append(f'   {name:28s}  - (no run)' + (f'   DIVERGED s{bad}' if bad else '')); continue
        g = lambda k: [v[k] for v in ok.values()]
        rel = g('rel')
        lines.append(f'   {name:28s} {len(ok):2d} {msd(g("unseen")):>13s} {msd(g("train")):>13s} '
                     + f'{np.mean(rel):.2f} [{min(rel):.2f}-{max(rel):.2f}]'.rjust(18)
                     + f'{np.mean(g("a3")):+10.1f} | {np.mean(g("a8")):+7.1f}   '
                     + ' '.join(f's{s} {v["unseen"]:.1f} ({v["rel"]:.2f})' for s, v in sorted(ok.items()))
                     + (f'   DIVERGED s{bad}' if bad else ''))
    for ref, rlab in (('Bfixkp20n', 'dose-normalised'), ('Bfixkp20', 'raw encoder'), ('ebvae', 'EB-VAE')):
        S = sorted(set(C['Bfixkp20nd']) & set(C[ref]))
        if S:
            du = np.array([C['Bfixkp20nd'][s]['unseen'] - C[ref][s]['unseen'] for s in S])
            dr = np.array([C['Bfixkp20nd'][s]['rel'] - C[ref][s]['rel'] for s in S])
            lines.append(f'   paired option (b) - {rlab:15s}: unseen {du.mean():+6.1f} pt (lower in {int((du < 0).sum())}/{len(S)} seeds), '
                         f'model / true {dr.mean():+.2f}')
    lines.append('')
open(f'{R}/encdoseval_table.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
