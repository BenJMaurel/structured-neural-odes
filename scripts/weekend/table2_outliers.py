#!/usr/bin/env python3
"""Outlier check for Table 2 with ten seeds .
Uses training data only to diagnose: each run's median signed AUC error on its OWN training patients (recal 'all' JSONs).
  (a) all ten seeds (as Table 2)
  (b) leaving out seed pairs in which either model's training-patient bias exceeds 10 % in absolute value (rule fixed here, after the
      runs, from training data only)
  (c) every run rescaled by its own training-patient level, per visit: pred / (1 + median signed error on its training patients),
      the recalibration of Supplementary Table S2
Per-seed outlier flags: |x - median| > 3 x 1.4826 MAD over the ten seeds. Paired patient bootstrap as in ndm_tables.py (seed-averaged
|err|, 2000 replicates). Writes results/s4_pnoise_win_more/table2_outliers.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, json
import numpy as np
os.chdir(REPO)
OLD, NEW, RO, RN = 'results/s4_pnoise_win/eval', 'results/s4_pnoise_win_more/eval', 'results/recal/paper_win', 'results/recal/paper_win_more'
SEEDS = list(range(1, 11))


def load(p, v):
    pp = json.load(open(p))['per_patient']
    return np.array(pp[f'pred_auc_{v}'], float), np.array(pp[f'true_auc_{v}'], float)


def runs(arch, v, recal=False):
    out = {}
    for s in SEEDS:
        p, t = load(f'{OLD if s <= 3 else NEW}/ep006000_{arch}_n100_s{s}_on_in.json', v)
        pt, tt = load(f'{RO if s <= 3 else RN}/{arch}_s{s}_all.json', v)
        b = float(np.median(pt / tt - 1))
        out[s] = dict(r=(p / (1 + b) if recal else p) / t - 1, train_bias=b)
    return out


def boot(da, db, rng):
    d = 100 * (np.median(da) - np.median(db)); n = len(da)
    bs = [100 * (np.median(da[i]) - np.median(db[i])) for i in (rng.integers(0, n, n) for _ in range(2000))]
    return d, *np.percentile(bs, [2.5, 97.5])


def flags(x):
    x = np.asarray(x); m = np.median(x); mad = 1.4826 * np.median(np.abs(x - m))
    return [SEEDS[i] for i in np.where(np.abs(x - m) > 3 * mad)[0]] if mad > 0 else []


L = [__doc__.split('\n')[0], '']
for v, lab in (('v1', 'V1 observed dose'), ('v2', 'V2 new dose 1-8 mg')):
    for recal in (False, True):
        F, D = runs('film', v, recal), runs('dc', v, recal)
        bad = [s for s in SEEDS if abs(F[s]['train_bias']) > 0.10 or abs(D[s]['train_bias']) > 0.10]
        for name, keep in (('all ten seeds', SEEDS), (f'without seed pairs {bad} (training-patient bias > 10 %)', [s for s in SEEDS if s not in bad])):
            if recal and name != 'all ten seeds':
                continue
            mf = [100 * np.median(np.abs(F[s]['r'])) for s in keep]; md = [100 * np.median(np.abs(D[s]['r'])) for s in keep]
            da = np.mean([np.abs(F[s]['r']) for s in keep], 0); db = np.mean([np.abs(D[s]['r']) for s in keep], 0)
            d, lo, hi = boot(da, db, np.random.default_rng(0))
            dd = np.array(mf) - np.array(md)
            L.append(f'{lab}, {"recalibrated on training patients" if recal else "as Table 2"}, {name}:')
            L.append(f'   OT-FiLM {np.mean(mf):.1f} +- {np.std(mf, ddof=1):.1f}   dose-cond {np.mean(md):.1f} +- {np.std(md, ddof=1):.1f}   '
                     f'paired {d:+.2f} [{lo:+.2f}, {hi:+.2f}]   OT-FiLM lower in {int((dd < 0).sum())}/{len(dd)}   median per-seed diff {np.median(dd):+.2f}')
            if name == 'all ten seeds':
                L.append(f'   per-seed OT-FiLM {" ".join(f"{x:.1f}" for x in mf)}   outlier flags {flags(mf)}')
                L.append(f'   per-seed dose-cond {" ".join(f"{x:.1f}" for x in md)}   outlier flags {flags(md)}')
                if not recal:
                    L.append(f'   training-patient bias % OT-FiLM {" ".join(f"{100 * F[s]["train_bias"]:+.1f}" for s in SEEDS)}')
                    L.append(f'   training-patient bias % dose-cond {" ".join(f"{100 * D[s]["train_bias"]:+.1f}" for s in SEEDS)}')
    L.append('')
open('results/s4_pnoise_win_more/table2_outliers.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
