#!/usr/bin/env python3
"""Recovery test of the a-posteriori physiology tracker: the null-test trackers on
confound_vc00_s4_pnoise_win_shift70 = the same 1100 test patients, doses and visit-1 data, with visit-2 Vmax x
exp(delta), delta ~ U(-0.7, 0.7), for half of them (the other half: exactly the null-test data).

True change per patient: Delta_true = log AUC at 4 mg (own formulation) with the visit-2 parameters minus with the
visit-1 ones (true simulator; truemodel.json g_true_v2 - g_true). A tracker 'flags' a change when |Delta_hat| > 1.96 sd.
Columns: false flags among UNSHIFTED patients; detection power among shifted patients by the size of the true change
in AUC at 4 mg; ROC AUC of |z| (shifted with |change| > 20 % vs unshifted); for shifted patients the slope of
Delta_hat on Delta_true (1 = unbiased, < 1 = shrunk towards no change), the correlation and the median |error|;
coverage of the 95 % interval over all patients. Network rows: mean over seeds 1-3.

    recovery_table.py [results/tracking/recovery] > results/tracking/recovery/recovery_table.txt
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

R = sys.argv[1] if len(sys.argv) > 1 else 'results/tracking/recovery'
COHORT = sys.argv[2] if len(sys.argv) > 2 else 'confound_vc00_s4_pnoise_win_shift70'
BINS = [(0.0, np.log(1.1), '<10%'), (np.log(1.1), np.log(1.25), '10-25%'), (np.log(1.25), np.log(1.5), '25-50%'),
        (np.log(1.5), 99.0, '>50%')]
KINDS = [('film', 'FiLM'), ('a', 'A (orth-split)'), ('b', 'B (dose history)')]
METHODS = [('enc', 'encoder means'), ('ind_eb', 'per-visit MAP, sigma eb'), ('jq0.5_eb', 'smoother q 0.5, sigma eb'),
           ('ind_ckpt', 'per-visit MAP, sigma 0.05'), ('jq0.5_ckpt', 'smoother q 0.5, sigma 0.05')]


def roc_auc(pos, neg):
    return float(mannwhitneyu(pos, neg, alternative='greater').statistic / (len(pos) * len(neg)))


def evaluate(dh, sd, dt, shifted, lbf=None):
    z = dh / sd
    flag = np.abs(z) > 1.96
    a = np.abs(dt)
    out = {'false_flags': 100 * np.mean(flag[~shifted])}
    for lo, hi, lab in BINS:
        m = shifted & (a >= lo) & (a < hi)
        out[f'power {lab}'] = 100 * np.mean(flag[m]) if m.any() else np.nan
    pos = shifted & (a > np.log(1.2))
    out['roc_auc'] = roc_auc(np.abs(z)[pos], np.abs(z)[~shifted])
    if lbf is not None:
        out['roc_auc_bf'] = roc_auc(lbf[pos], lbf[~shifted])
    s = shifted
    xc, yc = dt[s] - dt[s].mean(), dh[s] - dh[s].mean()
    out['slope'] = float((xc * yc).sum() / (xc ** 2).sum())
    out['corr'] = float(np.corrcoef(dt[s], dh[s])[0, 1])
    out['median_abs_err'] = 100 * float(np.median(np.abs(dh[s] - dt[s])))
    out['coverage95'] = 100 * float(np.mean(np.abs(dh - dt) < 1.96 * sd))
    return out


COLS = ['false_flags'] + [f'power {b[2]}' for b in BINS] + ['roc_auc', 'slope', 'corr', 'median_abs_err', 'coverage95']
FMT = {'roc_auc': '{:.2f}', 'slope': '{:.2f}', 'corr': '{:.2f}'}


def row(name, res):
    cells = []
    for c in COLS:
        v = [r[c] for r in res if c in r and np.isfinite(r[c])]
        f = FMT.get(c, '{:.1f}')
        cells.append((f.format(np.mean(v)) + ('' if len(v) < 2 else ' +- ' + f.format(np.std(v)))).rjust(15) if v else '-'.rjust(15))
    return f'{name:38s}' + ''.join(cells)


def main():
    truth = pd.read_csv(f'results/exp_film_run/{COHORT}/confound_truth.csv')
    truth['ID'] = truth['ID'].astype(int).astype(str)
    truth = truth.set_index('ID')
    tm = json.load(open(f'{R}/truemodel.json'))
    ids = tm['ids']
    dt_map = dict(zip(ids, np.array(tm['g_true_v2']) - np.array(tm['g_true'])))
    print(__doc__.strip().split('\n\n')[1].replace('\n', ' '))
    sh = truth.loc[ids, 'v2_shift_delta'].values != 0
    dt = np.array([dt_map[i] for i in ids])
    print(f'\n{len(ids)} test patients, {sh.sum()} shifted. True change in AUC at 4 mg among shifted, median |.| '
          f'{100 * np.median(np.abs(dt[sh])):.0f} %; patients per power bin '
          + ', '.join(f'{b[2]} {int(np.sum(sh & (np.abs(dt) >= b[0]) & (np.abs(dt) < b[1])))}' for b in BINS))
    a1, a2, a0 = (truth.loc[ids, c].values for c in ('AUC_v1', 'AUC_v2', 'AUC_v2_noshift'))
    print(f'visit-1 -> visit-2 AUC change among shifted (log): dose part sd {np.std(np.log(a0 / a1)[sh]):.2f}, '
          f'biology part sd {np.std(np.log(a2 / a0)[sh]):.2f}')
    print('\n' + f'{"":38s}' + ''.join(c.rjust(15) for c in COLS))
    print(row('TRUE MODEL  per-visit MAP-BE', [evaluate(np.array(tm['delta']), np.array(tm['sd']), dt, sh)]))
    for kind, kname in KINDS:
        runs = [json.load(open(f'{R}/{kind}_s{s}.json')) for s in (1, 2, 3) if os.path.exists(f'{R}/{kind}_s{s}.json')]
        if not runs:
            continue
        print(f'-- {kname}, seeds {len(runs)}')
        for mk, mname in METHODS:
            res, res_bf = [], []
            for r in runs:
                if mk not in r['methods']:
                    continue
                m = r['methods'][mk]
                pos = {i: k for k, i in enumerate(r['ids'])}
                idx = np.array([pos[i] for i in ids])
                dh, sd = np.array(m['delta'])[idx], np.array(m['sd'])[idx]
                lbf = np.array(m['log_bf_vs_j0'])[idx] if 'log_bf_vs_j0' in m else None
                res.append(evaluate(dh, sd, dt, sh, lbf))
            if res:
                print(row(f'   {mname}', res)
                      + (f'   ROC AUC with log BF {np.mean([r["roc_auc_bf"] for r in res]):.2f}' if 'roc_auc_bf' in res[0] else ''))


if __name__ == '__main__':
    main()
