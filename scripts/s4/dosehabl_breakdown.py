#!/usr/bin/env python3
"""Where does the dose-normalised encoder lose accuracy on the two-visit tacrolimus task?
Median |AUC error| (and signed median at V1) by visit-1 dose, dose ratio, formulation and exposure, for dh7pg / dh7pgkp / dh7pgen /
dh7pgkpn, seeds 1-2 averaged (in-range test cohort, ep6000). Eval JSON order = sorted test IDs (checked: 1083/1100 matched on the
true AUC pair, all in order). Second block: how much the visit doses reveal about the patient in this cohort (R2 of the log parameter
on the log dose; visit-1 doses are kept only if the visit-1 exposure is plausible, visit-2 doses are drawn independently).
Writes results/s4_doseh/dosehabl_breakdown.txt"""
import json
import numpy as np, pandas as pd

B = 'results/exp_film_run/confound_vc00_s4_pnoise_win'; E = 'results/s4_doseh/eval'
V = ('dh7pg', 'dh7pgkp', 'dh7pgen', 'dh7pgkpn'); SEEDS = (1, 2)
te = pd.read_csv(f'{B}/virtual_cohort_film_test.csv', low_memory=False)
v = te.groupby(['ID', 'VISIT']).agg(AUC=('AUC', 'first'), AMT=('AMT', 'first')).unstack()
tab = pd.DataFrame({'a1': v['AUC'][1], 'a2': v['AUC'][2], 'd1': v['AMT'][1], 'd2': v['AMT'][2]}).apply(pd.to_numeric)
tab = tab.join(pd.read_csv(f'{B}/confound_truth.csv').set_index('ID')[['CL_base', 'confound_par', 'DRUG']])


def errs(vn, s):
    q = json.load(open(f'{E}/ep006000_{vn}_osplit1-5_n100_s{s}_on_in.json'))['per_patient']
    t1, t2, p1, p2 = (np.array(q[k]) for k in ('true_auc_v1', 'true_auc_v2', 'pred_auc_v1', 'pred_auc_v2'))
    assert np.allclose(t1, tab.a1.values, rtol=1e-3) and np.allclose(t2, tab.a2.values, rtol=1e-3), 'eval order != sorted test IDs'
    return p1 / t1 - 1, p2 / t2 - 1


Er = {vn: [errs(vn, s) for s in SEEDS] for vn in V}
out = [__doc__.split('\n')[0], '']


def block(name, groups):
    out.append(f'== by {name}: median |err| V1 / V2 %, [signed median V1 %]')
    out.append(f'{"":13s}' + ''.join(f'{x:>22s}' for x in V) + '     n')
    for lab, m in groups:
        m = np.asarray(m)
        cells = []
        for vn in V:
            med = lambda j, f: np.mean([100 * f(e[j][m]) for e in Er[vn]])
            cells.append(f'{med(0, lambda x: np.median(np.abs(x))):5.1f}/{med(1, lambda x: np.median(np.abs(x))):5.1f} '
                         f'[{med(0, np.median):+5.1f}]'.rjust(22))
        out.append(f'{lab:13s}' + ''.join(cells) + f'  {m.sum():4d}')
    out.append('')


d1, d2 = tab.d1.values, tab.d2.values; r = np.log2(d2 / d1)
block('visit-1 dose (mg)', [(f'd1 {lo}-{hi}', (d1 >= lo) & (d1 <= hi)) for lo, hi in ((1, 2), (3, 4), (5, 6), (7, 8))])
block('log2(d2/d1)', [('< -1', r < -1), ('-1 .. 0', (r >= -1) & (r < 0)), ('0 .. 1', (r > 0) & (r <= 1)), ('> 1', r > 1)])
block('formulation', [(x, tab.DRUG.values == x) for x in ('Prograf', 'Advagraf')])
for lab, x in (('AUC1/d1', tab.a1 / tab.d1), ('AUC1', tab.a1)):
    q = np.quantile(x, [0, .25, .5, .75, 1])
    block(f'{lab} quartile', [(f'Q{k + 1}', (x >= q[k]) & (x <= q[k + 1])) for k in range(4)])


def r2(y, x):
    X = np.c_[np.ones(len(x)), x]; b = np.linalg.lstsq(X, y, rcond=None)[0]; return 1 - np.var(y - X @ b) / np.var(y), b[1]


out.append('== what the visit doses reveal (test cohort, 1100): R2 / slope of the log quantity on the log dose')
for lab, y, x in (('log CL_base ~ log d1', np.log(tab.CL_base), np.log(d1)), ('log CL_base ~ log d2', np.log(tab.CL_base), np.log(d2)),
                  ('log Vc ~ log d1', np.log(tab.confound_par), np.log(d1)), ('log Vc ~ log d2', np.log(tab.confound_par), np.log(d2)),
                  ('log(AUC1/d1) ~ log d1', np.log(tab.a1 / d1), np.log(d1)), ('log(AUC2/d2) ~ log d2', np.log(tab.a2 / d2), np.log(d2))):
    R, b = r2(np.asarray(y), np.asarray(x)); out.append(f'  {lab:24s} R2 {R:.2f}  slope {b:+.2f}')
out.append('  (d2 is independent of the patient, so the last slope is the average within-patient non-proportionality: AUC ~ d^(1+slope))')
open('results/s4_doseh/dosehabl_breakdown.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
