#!/usr/bin/env python3
"""Table of the tracker null test: results/tracking/null/{film,a,b}_s<seed>.json + truemodel.json.

Truth: the two visits have the SAME PK parameters, so every |Delta| is false movement. Delta = change of log AUC at the
reference regimen (4 mg, own formulation) between the V2 and the V1 state estimate. 'at d2' = the same comparison read
at the V2 dose (V2's own state vs V1's state carried to d2), which has no FiLM path inconsistency. Seed columns: mean
+- sd over seeds of each seed's statistic.
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, 'scripts/tracking')
from track_null import summarize

R = sys.argv[1] if len(sys.argv) > 1 else 'results/tracking/null'
KINDS = [('film', 'FiLM (Table 3 config)'), ('a', 'A: FiLM + orth-split 1/5/4'), ('b', 'B fixed: dose history (dh7pg)')]
METHODS = [('enc', 'ENC  encoder means'), ('ind_eb', 'IND  per-visit MAP, sigma eb'),
           ('j0_eb', 'J0   one state, sigma eb'), ('jq0.5_eb', 'Jq   smoother q 0.5, sigma eb'),
           ('ind_ckpt', 'IND  per-visit MAP, sigma 0.05'), ('jq0.5_ckpt', 'Jq   smoother q 0.5, sigma 0.05')]
BINS = [-np.inf, -1.0, -0.3, 0.3, 1.0, np.inf]


def ms(v, f='{:.1f}'):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return '   -   '
    return (f.format(np.mean(v)) + (' +- ' + f.replace('+', '').format(np.std(v)) if len(v) > 1 else '')).rjust(16)


def at_d2(m):
    return np.log(np.array(m['auc_fit_v2'])) - np.log(np.array(m['auc_cf_v2_from_v1']))


def binned(delta, x):
    idx = np.digitize(x, BINS[1:-1])
    return [100 * float(np.mean(delta[idx == b])) if np.any(idx == b) else np.nan for b in range(len(BINS) - 1)]


def main():
    tm = json.load(open(f'{R}/truemodel.json')) if os.path.exists(f'{R}/truemodel.json') else None
    print(__doc__.strip().split('\n\n')[1].replace('\n', ' '))
    print()
    hdr = (f'{"":34s}{"|D| median %":>16s}{"|D| p90 %":>16s}{"mean D %":>16s}{"slope/logDR":>16s}{"flag z>1.96":>16s}'
           f'{"P(|D|>20%)":>16s}{"BF>3":>16s}{"at d2 |D|":>16s}{"slope at d2":>16s}{"fit V1 %":>16s}{"cf V2 %":>16s}')
    if tm is not None:
        s = tm['summary']
        x = np.log(np.array(tm['dose_v2_mg']) / np.array(tm['dose_v1_mg']))
        d2 = np.log(np.array(tm['auc_fit_v2'])) - np.log(np.array(tm['auc_cf_v2_from_v1']))
        s2 = summarize(d2, x)
        print(f'-- TRUE MODEL (true structure + population + residual error), per-visit MAP-BE, n = {tm["n"]}')
        print(hdr)
        print(f'{"IND  per-visit MAP-BE":34s}' + ''.join(ms([v], f) for v, f in (
            (s['median_abs_pct'], '{:.1f}'), (s['p90_abs_pct'], '{:.1f}'), (s['mean_pct'], '{:+.1f}'),
            (s['slope_on_log_dose_ratio'], '{:+.3f}'), (100 * s['flag_z196'], '{:.1f}'), (100 * s['flag_p_gt20_095'], '{:.1f}'),
            (None, ''), (s2['median_abs_pct'], '{:.1f}'), (s2['slope_on_log_dose_ratio'], '{:+.3f}'),
            (s['auc_fit_v1_median_abs_err_pct'], '{:.1f}'), (s['auc_cf_v2_from_v1_median_abs_err_pct'], '{:.1f}'))))
        print(f'   mean D % by log(d2/d1) bin {[f"{b:g}" for b in BINS[1:-1]]}: '
              + ' / '.join(f'{v:+.1f}' for v in binned(np.array(tm['delta']), x)))
        print()
    for kind, kname in KINDS:
        files = sorted(glob.glob(f'{R}/{kind}_s[0-9]*.json'))
        if not files:
            continue
        runs = [json.load(open(f)) for f in files]
        seeds = [os.path.basename(f).split('_s')[1].split('.')[0] for f in files]
        sig = [r['methods'].get('ind_eb', {}).get('summary', {}).get('sigma') for r in runs]
        print(f'-- {kname}: seeds {seeds}, n = {runs[0]["n"]}, state dims {runs[0]["state_dims"]}, '
              f'sigma eb {", ".join(f"{v:.3f}" for v in sig if v)} (checkpoint 0.05)')
        print(hdr)
        for mk, mname in METHODS:
            got = [r['methods'][mk] for r in runs if mk in r['methods']]
            if not got:
                continue
            S = [m['summary'] for m in got]
            x = [np.log(np.array(r['dose_v2_mg']) / np.array(r['dose_v1_mg'])) for r in runs if mk in r['methods']]
            sd2 = [summarize(at_d2(m), xx) for m, xx in zip(got, x)]
            g = lambda k, scale=1.0: [scale * s[k] if k in s else None for s in S]
            print(f'{mname:34s}' + ms(g('median_abs_pct')) + ms(g('p90_abs_pct')) + ms(g('mean_pct'), '{:+.1f}')
                  + ms(g('slope_on_log_dose_ratio'), '{:+.3f}') + ms(g('flag_z196', 100)) + ms(g('flag_p_gt20_095', 100))
                  + ms(g('bf_gt3', 100)) + ms([s['median_abs_pct'] for s in sd2])
                  + ms([s['slope_on_log_dose_ratio'] for s in sd2], '{:+.3f}')
                  + ms(g('auc_fit_v1_median_abs_err_pct')) + ms(g('auc_cf_v2_from_v1_median_abs_err_pct')))
            if mk in ('ind_eb', 'jq0.5_eb'):
                bins = np.array([binned(np.array(m['delta']), xx) for m, xx in zip(got, x)])
                print(f'{"":6s}mean D % by log(d2/d1) bin {[f"{b:g}" for b in BINS[1:-1]]}: '
                      + ' / '.join(f'{v:+.1f}' for v in np.nanmean(bins, 0))
                      + f'; identity ratio median {np.mean(g("identity_ratio_median")):.3f}'
                      + (f'; displacement share FiLM / dyn / phys {np.mean(g("disp_frac_film_median")):.2f} / '
                         f'{np.mean(g("disp_frac_dyn_median")):.2f} / {np.mean(g("disp_frac_phys_median")):.2f}'
                         if kind == 'a' else ''))
            if mk.startswith('jq'):
                print(f'{"":6s}log BF (Jq vs J0) median {ms(g("log_bf_median"), "{:+.2f}").strip()}; '
                      f'BF > 10 {ms(g("bf_gt10", 100)).strip()} %')
            if mk.startswith('j0') and kind != 'b':
                print(f'{"":6s}(J0 |D| = FiLM path inconsistency T(d2->ref) o T(d1->d2) vs T(d1->ref); 0 by construction for B)')
        print()


if __name__ == '__main__':
    main()
