#!/usr/bin/env python3
"""Regimen benchmark of Baaz et al. (linear simulator, Table 4 of ndm.tex): does each model change a patient's exposure with the dose
by the right factor?. Same files and seeds as table4.py.
  1. WITHIN-PATIENT dose response: every test patient is predicted at 350 and at 850 mg (training schedule); the model's ratio
     AUC(850)/AUC(350) for that patient vs the true one (median over the 400 test patients, per seed; true ratio ~2.38, slightly below
     850/350 because patients start with drug on board). The true per-patient AUCs come from the EB-VAE files of the same seed
     (auc_pred / (1 + auc_err)); test patients are identical across models and across randomised / titrated training.
  2. ARM SPLIT: median signed AUC error by the arm the test patient was OBSERVED on (test arms alternate in both runners, also for
     titrated training: even index 400 mg, odd 800 mg), on the training regimen and in the two cross-arm cells (observed on 800 mg ->
     predicted at 350 mg; observed on 400 -> 850). A dose response that is too flat shows as over-prediction in the first and
     under-prediction in the second.
  3. The true exposure ratio between the two TRAINING arms (geometric mean AUC of the 800-mg arm / the 400-mg arm): 1.97 when randomised;
     under titration the high dose goes to fast eliminators, and the ratio is what a model that takes the arm difference for the dose
     effect would learn.
Writes results/regimen_dh/baaz_arm_split.txt."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
R = os.path.join(ROOT, 'results/regimen_dh'); EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'
SEEDS = {'sparse': range(5), 'titr': range(3), 'dense': range(3)}
UNSEEN = ('amt350', 'amt600', 'amt850', 'time400', 'time800')
MODELS = (('ebvae', 'EB-VAE'), ('Bfixkp20n', 'dose-event (ours)'), ('dcn', 'dose-cond field'))


def path(model, col, s):          # as table4.py
    d = 'dense' if col == 'dense' else 'sparse'; tag = f'N100_{d}_s{s}.json'
    if model == 'ebvae':
        p = f'{EB}/res_titr/{"titr0.8" if col == "titr" else "ctrl"}/{tag}'
        return p if os.path.exists(p) or col == 'titr' else f'{EB}/res/N100_{d}' + (f'_s{s}' if s else '') + '.json'
    if model == 'dcn':
        return f'{R}/dosecond_n/{"titr0.8/" if col == "titr" else ""}{tag}'
    return f'{R}/{"titr0.8/" if col == "titr" else ""}{model}/{tag}'


def truth(col, s):
    r = json.load(open(path('ebvae', col, s)))['individual']
    return {g: np.array(r[g]['auc_pred_per_patient']) / (1 + np.array(r[g]['auc_err_per_patient'])) for g in ('amt350', 'amt850')}


def ms(v):
    return f'{np.mean(v):.2f} [{np.min(v):.2f}-{np.max(v):.2f}]'


lines = [__doc__.strip().split('\n')[0], '']
for col in ('sparse', 'titr', 'dense'):
    lab_col = {'sparse': 'sparse, randomised arms', 'titr': 'sparse, TITRATED arms', 'dense': 'dense, randomised arms'}[col]
    lines.append(f'=== linear simulator, {lab_col}')
    tr_ratio = [np.median(truth(col, s)['amt850'] / truth(col, s)['amt350']) for s in SEEDS[col]]
    lines.append(f'   true within-patient AUC(850)/AUC(350): median {np.mean(tr_ratio):.2f}')
    lines.append(f'   {"model":18s} {"n":>2s}  {"model 850:350 ratio":>22s} {"model / true":>22s}   {"own regimen 400|800":>19s}   '
                 f'{"800->350 | 400->850":>19s}   unseen err per seed')
    for m, lab in MODELS:
        rat, rel, own, cross, un = [], [], [], [], []
        for s in SEEDS[col]:
            p = path(m, col, s)
            if not os.path.exists(p):
                continue
            r = json.load(open(p))['individual']
            u = float(np.mean([r[g]['auc_abs_median'] for g in UNSEEN]))
            if not np.isfinite(u) or u > 1000:
                continue
            e = {g: np.array(r[g]['auc_err_per_patient']) for g in ('train', 'amt350', 'amt850')}
            t = truth(col, s)
            q = (1 + e['amt850']) / (1 + e['amt350'])                     # model ratio / true ratio, per patient
            rel.append(np.median(q)); rat.append(np.median(q * t['amt850'] / t['amt350']))
            own.append((100 * np.median(e['train'][0::2]), 100 * np.median(e['train'][1::2])))
            cross.append((100 * np.median(e['amt350'][1::2]), 100 * np.median(e['amt850'][0::2])))
            un.append(f's{s} {u:.1f} (rel {np.median(q):.2f})')
        if not rel:
            continue
        lines.append(f'   {lab:18s} {len(rel):2d}  {ms(rat):>22s} {ms(rel):>22s}   {np.mean([o[0] for o in own]):+7.1f}|{np.mean([o[1] for o in own]):+7.1f}   '
                     f'{np.mean([c[0] for c in cross]):+7.1f}|{np.mean([c[1] for c in cross]):+7.1f}   ' + '  '.join(un))
    if col != 'dense':          # true exposure ratio between the training arms
        sys.path.insert(0, EB); import gen_regimens as G
        out = []
        for s in SEEDS[col]:
            rng = np.random.default_rng(s); P = G.draw(200, rng)
            arm = G.titrate(P, 0.8, s)[0] if col == 'titr' else np.where(np.arange(200) % 2 == 0, 400.0, 800.0)
            t, c = G.simulate(P, [0, 8, 16, 24], np.repeat(arm[:, None], 4, 1)); a = np.trapezoid(c, t, axis=1)
            out.append(np.exp(np.mean(np.log(a[arm == 800])) - np.mean(np.log(a[arm == 400]))))
        lines.append(f'   true exposure ratio between the TRAINING arms (geometric mean AUC, 800-mg arm / 400-mg arm), per seed: '
                     + ' '.join(f'{v:.2f}' for v in out))
    lines.append('')
lines.append('model 850:350 ratio: median over test patients of the model\'s AUC(850 mg) / AUC(350 mg) for the same patient; model / true: '
             'the same divided by the true ratio (1 = right; mean [min-max] over seeds). Own regimen / cross-arm: median signed AUC error (%) '
             'by the arm the patient was observed on, mean over seeds. Diverged seeds (unseen > 1000 %) left out.')
open(f'{R}/baaz_arm_split.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
