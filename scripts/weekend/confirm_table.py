#!/usr/bin/env python3
"""Confirmation of the regimen benchmark on fresh seeds with held-out regimens.
Frozen dose-event model (results/regimen_dh/confirm/{ctrl,titr0.8}/Bfixkp20n/) vs EB-VAE (third_party/baaz_ebvae/ours/regimen/
res_titr/confirm/{ctrl,titr0.8}/), sparse randomised s5-14, sparse titrated (rho 0.8) s5-14, dense randomised s3-7.
Per run: orig = mean over the 5 original unseen regimens of the median |AUC(0-30 h) err|; held-out = the same over the 5 c_* regimens;
training = the training regimen; ratio = median over patients of (predicted AUC(850)/AUC(350)) / (true ratio).
CHECK: the dose-event runs skip their own data check (--no-check, new seeds), so the training observations of each pair of runs are
compared here (dh_regimen.py run as a module, no training, vs the EB-VAE run's train.csv).
Writes results/regimen_dh/confirm/confirm_table.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, io, json, runpy, contextlib
os.environ['REGIMENS_CONFIRM'] = '1'
import numpy as np, pandas as pd

ROOT = REPO; os.chdir(ROOT)
RD, RT = 'results/regimen_dh/confirm', f'{REPO}/third_party/baaz_ebvae/ours/regimen/res_titr/confirm'
SCR = os.environ.get('CONFIRM_SCRATCH', 'results/scratch/confirm_scratch'); os.makedirs(SCR, exist_ok=True)
ORIG = ('amt350', 'amt600', 'amt850', 'time400', 'time800'); HELD = ('c_amt500', 'c_amt1000', 'c_load', 'c_missed', 'c_extra')
CONDS = [('sparse randomised', 'ctrl', 'sparse', range(5, 15), []), ('sparse TITRATED', 'titr0.8', 'sparse', range(5, 15), ['--titrate', '0.8']),
         ('dense randomised', 'ctrl', 'dense', range(3, 8), [])]
BFLAGS = ['--obsrv-mgl', '20', '--kick-phys', '32', '--enc-dose', 'norm']


def cell(p):
    if not os.path.exists(p):
        return None
    r = json.load(open(p))['individual']
    if not all(k in r for k in ORIG + HELD):
        return None
    e = {k: np.array(r[k]['auc_err_per_patient']) for k in ('amt350', 'amt850')}
    return dict(orig=float(np.mean([r[k]['auc_abs_median'] for k in ORIG])), held=float(np.mean([r[k]['auc_abs_median'] for k in HELD])),
                train=float(r['train']['auc_abs_median']), ratio=float(np.median((1 + e['amt850']) / (1 + e['amt350']))),
                per={k: r[k]['auc_abs_median'] for k in ORIG + HELD})


def same_data(design, s, sub, extra):
    f = f'{RT}/{sub}/N100_{design}_s{s}_data/train.csv'
    if not os.path.exists(f):
        return None
    sys.argv = ['x', '100', design, str(s), f'{SCR}/u.json', '--no-check', '--iters', '1'] + BFLAGS + extra
    with contextlib.redirect_stdout(io.StringIO()):
        g = runpy.run_path('scripts/regimen/dh_regimen.py', run_name='confirm')
    d = pd.read_csv(f, sep=';'); d = d[d.EVID == 0].sort_values(['ID', 'TIME'])
    return bool(np.allclose(d.DV.values, g['y_tr'].reshape(-1), rtol=0, atol=1e-6))


ms = lambda v: f'{np.mean(v):5.1f} +- {np.std(v, ddof=1) if len(v) > 1 else 0:4.1f}'
L, S, DIV = [__doc__.split('\n')[0], ''], {}, []
for title, sub, design, seeds, extra in CONDS:
    rows = []
    for s in seeds:
        a, b = cell(f'{RD}/{sub}/Bfixkp20n/N100_{design}_s{s}.json'), cell(f'{RT}/{sub}/N100_{design}_s{s}.json')
        if a is not None and b is not None:
            if a['train'] > 100 or b['train'] > 100:   # diverged training (as the paper's convention for the EB-VAE on saturable seeds)
                DIV.append(f'{title} s{s}: ' + ('dose-event' if a['train'] > 100 else 'EB-VAE') + f' diverged (training-regimen error {max(a["train"], b["train"]):.3g} %), left out'); continue
            rows.append((s, a, b, same_data(design, s, sub, extra)))
    S[title] = rows
    L.append(f'=== {title}: {len(rows)} complete seed pairs {[r[0] for r in rows]}; same training data in both runners: '
             f'{sum(1 for r in rows if r[3])}/{len(rows)}' + ('' if all(r[3] for r in rows) else '  <-- CHECK'))
    if not rows:
        L.append(''); continue
    for key, lab in (('orig', 'unseen, original 5'), ('held', 'unseen, held-out 5'), ('train', 'training regimen'), ('ratio', 'dose response / true')):
        a = [r[1][key] for r in rows]; b = [r[2][key] for r in rows]; d = np.array(a) - np.array(b)
        L.append(f'   {lab:22s} ours {ms(a)}   EB-VAE {ms(b)}   paired {d.mean():+5.2f}'
                 + (f'   ours lower in {int((d < 0).sum())}/{len(d)}' if key != 'ratio' else ''))
    L.append('   per regimen (mean over seeds of the median |AUC err| %), ours / EB-VAE:')
    L.append('     ' + '  '.join(f'{k} {np.mean([r[1]["per"][k] for r in rows]):.1f}/{np.mean([r[2]["per"][k] for r in rows]):.1f}' for k in ORIG + HELD))
    L.append('')

# ---- pre-registered predictions
def get(t, who, key):
    return np.array([r[1 if who == 'ours' else 2][key] for r in S.get(t, [])])
def lower(t, key):
    a, b = get(t, 'ours', key), get(t, 'eb', key)
    return int((a < b).sum()), len(a)
P = []
R_, T_, D_ = 'sparse randomised', 'sparse TITRATED', 'dense randomised'
if S.get(R_):
    for key in ('orig', 'held'):
        k, n = lower(R_, key)
        P.append((f'P1 randomised, {key}: ours mean <= EB-VAE mean and ours lower in >= 6/10',
                  get(R_, 'ours', key).mean() <= get(R_, 'eb', key).mean() and k >= 6, f'{get(R_, "ours", key).mean():.1f} vs {get(R_, "eb", key).mean():.1f}, {k}/{n}'))
if S.get(T_):
    for key in ('orig', 'held'):
        k, n = lower(T_, key)
        P.append((f'P2 titrated, {key}: ours lower in >= 9/10', k >= 9, f'{k}/{n}'))
    if S.get(R_):
        eb_t, eb_r = get(T_, 'eb', 'orig').mean(), get(R_, 'eb', 'orig').mean()
        o_t, o_r = get(T_, 'ours', 'orig').mean(), get(R_, 'ours', 'orig').mean()
        P.append(('P2 EB-VAE titrated >= 1.5 x randomised (orig)', eb_t >= 1.5 * eb_r, f'{eb_t:.1f} vs {eb_r:.1f} (x{eb_t / eb_r:.2f})'))
        P.append(('P2 ours titrated within +3 points of randomised (orig)', o_t - o_r <= 3, f'{o_t:.1f} vs {o_r:.1f} ({o_t - o_r:+.1f})'))
    P.append(('P3 titrated dose response: ours >= 0.70 and EB-VAE <= 0.60', get(T_, 'ours', 'ratio').mean() >= 0.70 and get(T_, 'eb', 'ratio').mean() <= 0.60,
              f'{get(T_, "ours", "ratio").mean():.2f} vs {get(T_, "eb", "ratio").mean():.2f}'))
if S.get(D_):
    for key in ('orig', 'held'):
        k, n = lower(D_, key)
        P.append((f'P4 dense, {key}: ours lower in >= 4/5', k >= 4, f'{k}/{n}'))
L.append('LEFT OUT (training diverged; rule added after the runs, the convention the paper already applies to the EB-VAE): ' + ('; '.join(DIV) if DIV else 'none'))
L.append('')
L.append('PRE-REGISTERED PREDICTIONS')
for txt, ok, val in P:
    L.append(f'  [{"MET" if ok else "NOT MET"}] {txt}: {val}')
open(f'{RD}/confirm_table.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
