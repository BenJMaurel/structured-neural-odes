#!/usr/bin/env python3
"""A's physiology block as the tracking state: side-by-side null test and recovery test
of FiLM, A (full 10-D state), A physiology-only (track_null.py --phys-only: shared non-physiology state, 4-D physiology
change w), B and the true model. Network rows: mean +- sd over seeds 1-3.

A phys rows come twice: 'path' = Delta g_ref(s2, d2) - g_ref(s1, d1), the definition of every other row (includes the
FiLM path inconsistency that J0 measures, which the posterior sd does not cover); 'clean' = the change of AUC at 4 mg
caused by the physiology change alone (0 when w = 0).

    phys_table.py > results/tracking/phys_table.txt
"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, 'scripts/tracking')
from recovery_table import evaluate, BINS

ROWS = [('film', 'ind', 'FiLM     IND  per-visit MAP'), ('film', 'jq0.5', 'FiLM     Jq   smoother'),
        ('a', 'ind', 'A full   IND  per-visit MAP'), ('a', 'j0', 'A full   J0   (path inconsistency only)'),
        ('a', 'jq0.5', 'A full   Jq   smoother'),
        ('aphys', 'indp', 'A phys   INDp per-visit physiology, path'), ('aphys', 'indpc', 'A phys   INDp per-visit physiology, clean'),
        ('aphys', 'jpq0.5', 'A phys   JPq  smoother, path'), ('aphys', 'jpq0.5c', 'A phys   JPq  smoother, clean'),
        ('b', 'ind', 'B        IND  per-visit MAP'), ('b', 'jq0.5', 'B        Jq   smoother')]


def runs(R, kind):
    return [json.load(open(f'{R}/{kind}_s{s}.json')) for s in (1, 2, 3) if os.path.exists(f'{R}/{kind}_s{s}.json')]


def ms(v, f='{:.1f}', w=15):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return '-'.rjust(w)
    return (f.format(np.mean(v)) + (' +- ' + f.replace('+', '').format(np.std(v)) if len(v) > 1 else '')).rjust(w)


def null_block(sig):
    R = 'results/tracking/null'
    cols = [('median_abs_pct', '|D| med %', 1, '{:.1f}'), ('p90_abs_pct', '|D| p90 %', 1, '{:.1f}'),
            ('slope_on_log_dose_ratio', 'dose leak', 1, '{:+.3f}'), ('flag_z196', 'flags z %', 100, '{:.1f}'),
            ('z_sd', 'z sd', 1, '{:.2f}'), ('bf_gt3', 'BF>3 %', 100, '{:.1f}'),
            ('auc_fit_v1_median_abs_err_pct', 'fit V1 %', 1, '{:.1f}'), ('auc_cf_v2_from_v1_median_abs_err_pct', 'cf V2 %', 1, '{:.1f}')]
    print(f'\n== NULL TEST (same PK parameters at both visits: every |Delta| is false movement), sigma {sig}')
    print(f'{"":44s}' + ''.join(c[1].rjust(15) for c in cols))
    tm = json.load(open(f'{R}/truemodel.json')) if os.path.exists(f'{R}/truemodel.json') else None
    if tm is not None and sig == 'eb':
        s = tm['summary']
        print(f'{"TRUE MODEL  per-visit MAP-BE":44s}' + ''.join(ms([s.get(k) * m if s.get(k) is not None else None], f)
                                                             for k, _, m, f in cols))
    for kind, mk, name in ROWS:
        got = [r['methods'][f'{mk}_{sig}'] for r in runs(R, kind) if f'{mk}_{sig}' in r['methods']]
        if got:
            S = [g['summary'] for g in got]
            print(f'{name:44s}' + ''.join(ms([s[k] * m if k in s else None for s in S], f) for k, _, m, f in cols))


def recovery_block(sig):
    R, C = 'results/tracking/recovery', 'confound_vc00_s4_pnoise_win_shift70'
    truth = pd.read_csv(f'results/exp_film_run/{C}/confound_truth.csv')
    truth['ID'] = truth['ID'].astype(int).astype(str)
    truth = truth.set_index('ID')
    tm = json.load(open(f'{R}/truemodel.json'))
    ids = tm['ids']
    sh = truth.loc[ids, 'v2_shift_delta'].values != 0
    dt = np.array(tm['g_true_v2']) - np.array(tm['g_true'])
    cols = [('false_flags', 'false flags %', '{:.1f}')] + [(f'power {b[2]}', f'power {b[2]}', '{:.0f}') for b in BINS] + [
        ('roc_auc', 'ROC |z|', '{:.2f}'), ('roc_auc_bf', 'ROC BF', '{:.2f}'), ('slope', 'slope', '{:.2f}'),
        ('corr', 'corr', '{:.2f}'), ('coverage95', 'cover95 %', '{:.0f}')]
    print(f'\n== RECOVERY TEST (visit-2 Vmax shifted for {sh.sum()} of {len(ids)} patients; flag = |Delta| > 1.96 sd), sigma {sig}')
    print(f'{"":44s}' + ''.join(c[1].rjust(15) for c in cols))
    if sig == 'eb':
        e = evaluate(np.array(tm['delta']), np.array(tm['sd']), dt, sh)
        print(f'{"TRUE MODEL  per-visit MAP-BE":44s}' + ''.join(ms([e.get(k)], f) for k, _, f in cols))
    for kind, mk, name in ROWS:
        res = []
        for r in runs(R, kind):
            m = r['methods'].get(f'{mk}_{sig}')
            if m is None or 'sd' not in m:
                continue
            pos = {i: k for k, i in enumerate(r['ids'])}
            idx = np.array([pos[i] for i in ids])
            lbf = np.array(m['log_bf_vs_j0'])[idx] if 'log_bf_vs_j0' in m else None
            res.append(evaluate(np.array(m['delta'])[idx], np.array(m['sd'])[idx], dt, sh, lbf))
        if res:
            print(f'{name:44s}' + ''.join(ms([e.get(k) for e in res], f) for k, _, f in cols))


def main():
    print(__doc__.strip().split('\n\n')[0].replace('\n', ' '))
    print(__doc__.strip().split('\n\n')[1].replace('\n', ' '))
    for sig in ('eb', 'ckpt'):
        null_block(sig)
    for sig in ('eb', 'ckpt'):
        recovery_block(sig)
    ap = runs('results/tracking/null', 'aphys')
    if ap:
        print(f'\nA phys checks: physiology prior dose gap {max(r["phys_prior_dose_gap"] for r in ap):.1e}; max |p2 - p1 - w| '
              + f'{max(m["summary"]["phys_conservation_max"] for r in ap for k, m in r["methods"].items() if "phys_conservation_max" in m["summary"]):.1e}'
              + '; sigma eb ' + ', '.join(f'{r["methods"]["j0_eb"]["summary"]["sigma"]:.4f}' for r in ap))
        a = runs('results/tracking/null', 'a')
        if a:
            print('   (A full sigma eb ' + ', '.join(f'{r["methods"]["ind_eb"]["summary"]["sigma"]:.4f}' for r in a) + ')')


if __name__ == '__main__':
    main()
