#!/usr/bin/env python3
"""Why does the EB-VAE lose on our tacrolimus benchmarks?. Three checks:
 1. training vs test patients, saved models (eval_saved.py, re-scoring reproduces each run's own test JSON exactly): capacity
    limit (training patients fitted as badly as test ones) or generalisation gap (training patients fitted well)?
 2. scenario 3 with a bigger bottleneck (2 IC + 4 dynamic latents, 4-D state; published 1, 2, 2)
 3. scenario 3 with the noise FIXED at the simulator's true level (closest combined-2 fit to sd = 0.71 + 0.113 C; the published
    schedule starts at add 0.1 / prop 0.01 and learns them after 200 epochs, ending far below the truth)
Errors: AUC of the observed interval (scenario 3, factual) or of visits 1 / 2 (Part 2); per-run median |err| and RMSPE, mean +- sd
over runs; paired over the same 10 cohorts.

    ebcheck_table.py > results/baaz_check/ebcheck_table.txt
"""
import os, re, glob, json
import numpy as np, pandas as pd
from scipy import stats

T = 'results/baaz_check/trainvstest'
VAR = [('pub', 'runs', 'EB-VAE published (1,2,2; learned noise)'), ('big', 'runs_big', 'EB-VAE bigger bottleneck (2,4,4)'),
       ('true', 'runs_truenoise', 'EB-VAE noise fixed at truth')]


def err(pp, v=''):
    k = f'_{v}' if v else ''
    return np.array(pp[f'pred_auc{k}']) / np.array(pp[f'true_auc{k}']) - 1


def ms(x, f='{:.1f}'):
    x = [v for v in x if np.isfinite(v)]
    return (f.format(np.mean(x)) + ' +- ' + f.format(np.std(x, ddof=1) if len(x) > 1 else 0)) if x else '-'


def main():
    print(__doc__.strip().split('\n\n')[0])
    # ---------------------------------------------------------------- 1. Part 2
    print('\n== 1a. Part 2 dose-switch cohort, published EB-VAE: training patients (100; their visit 2 WAS in the loss) vs test (1261)')
    print(f'   {"seed / epoch":14s} {"V1 train":>9s} {"V1 test":>8s} {"V2 train":>9s} {"V2 test":>8s}   (median |AUC err| %; RMSPE in brackets)')
    for s in (1, 2, 3):
        for E in (300, 1000):
            f = {w: f'{T}/part2_s{s}_ep{E}_on_{w}.json' for w in ('train', 'test_in')}
            if not all(os.path.exists(p) for p in f.values()):
                continue
            J = {w: json.load(open(p))['summary'] for w, p in f.items()}
            c = lambda w, v: f'{J[w][v]["median_abs"]:5.1f} ({J[w][v]["rmspe"]:5.1f})'
            print(f'   s{s} ep{E:<5d}     {c("train", "v1")} {c("test_in", "v1")} {c("train", "v2")} {c("test_in", "v2")}')
    # ---------------------------------------------------------------- 1b. scenario 3 train vs test for every variant
    print('\n== 1b. Scenario 3, training (160) vs test (40) patients, per-run median |AUC err| / RMSPE, mean +- sd over runs')
    for key, d, name in VAR:
        pref = 'scen3' if key == 'pub' else f'scen3{key}'
        for E in (300, 1000):
            tr, te, n = [], [], 0
            for s in range(1, 11):
                ft, fe = f'{T}/{pref}_s{s}_ep{E}_on_train.json', f'{T}/{pref}_s{s}_ep{E}_on_test.json'
                if not (os.path.exists(ft) and os.path.exists(fe)):
                    continue
                a, b = json.load(open(ft))['summary']['factual'], json.load(open(fe))
                own = f'results/baaz_scen3/{d}/eval_test_s{s}_seed{s}_ep{E}.json'
                if os.path.exists(own):   # the re-scored test set must equal the run's own evaluation
                    o = json.load(open(own))['per_patient']['pred_auc']
                    assert np.max(np.abs(np.array(o) - np.array(b['per_patient']['pred_auc']))) < 1e-6, (key, s, E)
                b = b['summary']['factual']
                tr.append((a['median_abs'], a['rmspe'])); te.append((b['median_abs'], b['rmspe'])); n += 1
            if n:
                tr, te = np.array(tr), np.array(te)
                print(f'   {name:42s} ep{E:<5d} n {n:2d}  train {ms(tr[:, 0])} / {ms(tr[:, 1])}   test {ms(te[:, 0])} / {ms(te[:, 1])}'
                      f'   gap (test - train) median {ms(te[:, 0] - tr[:, 0])} pt')
    # ---------------------------------------------------------------- 2-3. scenario 3 test error by variant and budget
    print('\n== 2-3. Scenario 3 test error by variant and stage-2 budget (10 cohorts, 40 test patients each)')
    rows = {}
    for s in range(1, 11):
        Rr = f'results/scen_rerun/runs/s3_seed{s:03d}'
        lode = json.load(open(f'{Rr}/lode_final.json'))['per_series']
        L = pd.Series(np.array(lode['pred_auc']) / np.array(lode['true_auc']) - 1, index=[int(i) for i in lode['id']])
        x = pd.read_csv(f'{Rr}/tacro_mapbayest_auc_sigvar.csv').set_index('ID')
        rows.setdefault('Latent ODE, final ckpt', {})[s] = L
        rows.setdefault('MAP-BE (SIGMA fixed)', {})[s] = x.auc_ipred / x.AUC_observed - 1
        for key, d, name in VAR:
            for f in glob.glob(f'results/baaz_scen3/{d}/eval_test_s{s}_seed{s}_ep*.json'):
                E = int(re.search(r'_ep(\d+)', f).group(1)); pp = json.load(open(f))['per_patient']
                rows.setdefault(f'{name}, {E} ep', {})[s] = pd.Series(err(pp), index=pp['id'])
    stat = lambda v: (100 * np.median(np.abs(v)), 100 * np.sqrt(np.mean(v ** 2)))
    keys = ['Latent ODE, final ckpt', 'MAP-BE (SIGMA fixed)'] + sorted([k for k in rows if k.startswith('EB-VAE')],
                                                                         key=lambda k: ([v[2] for v in VAR].index(k.rsplit(', ', 1)[0]), int(k.rsplit(', ', 1)[1].split()[0])))
    print(f'   {"method":52s} runs   median |err|      RMSPE')
    for k in keys:
        v = [stat(e.values) for e in rows[k].values()]
        print(f'   {k:52s} {len(v):4d}   {ms([a for a, _ in v]):>13s}   {ms([b for _, b in v]):>13s}')
    print('   paired over cohorts (variant minus comparator; per-run median |err| and RMSPE; Wilcoxon):')
    for key, d, name in VAR[1:]:
        for E in (300, 1000):
            k = f'{name}, {E} ep'
            if k not in rows:
                continue
            for c in (f'{VAR[0][2]}, {E} ep', 'Latent ODE, final ckpt', 'MAP-BE (SIGMA fixed)'):
                S = sorted(set(rows[k]) & set(rows.get(c, {})))
                if len(S) < 5:
                    continue
                for j, lab in ((0, 'median|err|'), (1, 'RMSPE')):
                    dd = np.array([stat(rows[k][s].values)[j] - stat(rows[c][s].values)[j] for s in S])
                    p = stats.wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
                    print(f'     {k:42s} vs {c:46s} {lab:11s} {np.mean(dd):+6.2f} pt (better in {int((dd < 0).sum())}/{len(S)}, p {p:.2g})')


if __name__ == '__main__':
    main()
