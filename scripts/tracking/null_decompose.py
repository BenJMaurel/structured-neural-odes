#!/usr/bin/env python3
"""Split the tracker's false movement into sampling noise and model error.

Per patient (paired by ID), the true model's per-visit MAP-BE Delta (results/tracking/null/truemodel.json) is the false
movement ANY correct model would show from the same noisy 0/1/3 h samples. For each network run and method, regress
Delta_net on Delta_true: the share of Var(Delta_net) it explains is noise; the residual (Delta_net - b * Delta_true) is
model error, and its slope on log(d2/d1) is the dose-map part of it. Same readout in both: log AUC at 4 mg, own
formulation (FiLM / A transport to 4 mg with their FiLM; B and the true model simulate the dosing history at 4 mg).

    null_decompose.py [results/tracking/null] > results/tracking/null/null_decompose.txt
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, 'scripts/tracking')
from track_null import ols

R = sys.argv[1] if len(sys.argv) > 1 else 'results/tracking/null'


def main():
    tm = json.load(open(f'{R}/truemodel.json'))
    dt = dict(zip(tm['ids'], tm['delta']))
    gt = dict(zip(tm['ids'], tm['g_true']))
    print(__doc__.strip().split('\n\n')[1].replace('\n', ' '))
    print(f'\ntrue model: n {tm["n"]}, median |Delta| {100 * np.median(np.abs(tm["delta"])):.1f} %, '
          f'sd {100 * np.std(tm["delta"]):.1f} %\n')
    print(f'{"run":10s}{"method":14s}{"n":>6s}{"sd net %":>10s}{"sd true %":>10s}{"corr":>7s}{"b":>7s}{"R2 noise":>9s}'
          f'{"sd excess %":>12s}{"|excess| med %":>15s}{"excess slope":>13s}{"V1 readout err %":>17s}')
    for f in sorted(glob.glob(f'{R}/*_s[0-9]*.json')):
        r = json.load(open(f))
        run = os.path.basename(f)[:-5]
        x_all = np.log(np.array(r['dose_v2_mg']) / np.array(r['dose_v1_mg']))
        keep = np.array([i in dt for i in r['ids']])
        ids = [i for i in r['ids'] if i in dt]
        x = x_all[keep]
        d_true = np.array([dt[i] for i in ids])
        g_tr = np.array([gt[i] for i in ids])
        for mk in ('enc', 'ind_eb', 'jq0.5_eb', 'ind_ckpt'):
            if mk not in r['methods']:
                continue
            m = r['methods'][mk]
            d = np.array(m['delta'])[keep]
            c = float(np.corrcoef(d, d_true)[0, 1])
            b, _ = ols(d_true, d)
            ex = d - b * d_true
            bs, se = ols(x, ex)
            v1err = 100 * np.median(np.abs(np.array(m['g1'])[keep] - g_tr))
            print(f'{run:10s}{mk:14s}{len(d):6d}{100 * np.std(d):10.1f}{100 * np.std(d_true):10.1f}{c:7.2f}{b:7.2f}'
                  f'{c ** 2:9.2f}{100 * np.std(ex):12.1f}{100 * np.median(np.abs(d - d_true)):15.1f}'
                  f'{bs:+9.3f}+-{se:.3f}{v1err:13.1f}')
    print('\nV1 readout err = median |log AUC at 4 mg from the V1 state - truth| (true parameters), i.e. how well one '
          'visit of 3 samples pins the physiology readout.')


if __name__ == '__main__':
    main()
