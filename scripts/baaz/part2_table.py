#!/usr/bin/env python3
"""Part 2 table: EB-VAE (Baaz 2026 + patches) vs our OT-FiLM on the dose-switch cohort, 3 seeds each, ep6000 / final.
Metrics as Table 3 (ndm_tables.metrics): median |AUC err|, bias, within 20 %, RMSPE w/o worst 1 %, n > 100 %, per visit
(V1 observed dose; V2 = 4 days after the switch, in / above / below range). Paired: seed-averaged |err| per patient,
patients matched across pipelines on the true AUC (one-to-one check), median difference + Wilcoxon."""
import os, sys, glob, json, numpy as np
sys.argv = sys.argv[:1]; sys.path.insert(0, 'scripts/ndm')
from ndm_tables import metrics
from scipy.stats import wilcoxon
R = 'results/baaz_part2'
def load(paths, vis):
    out = []
    for p in paths:
        if not os.path.exists(p): continue
        j = json.load(open(p))['per_patient']; out.append((np.array(j[f'true_auc_{vis}']), np.array(j[f'pred_auc_{vis}'])))
    return out
def summary(runs):
    ms = [metrics(t, p) for t, p in runs]
    return {k: (np.mean([m[k] for m in ms]), [m[k] for m in ms]) for k in ('med', 'bias', 'w20', 'trm', 'n100')}
for reg, lab in (('in', 'in range 1-8 mg'), ('hi', 'above range 10-12 mg'), ('lo', 'below range 0.25-0.5 mg')):
    for vis in (('v1', 'v2') if reg == 'in' else ('v2',)):
        print(f'-- {"V1 (observed dose)" if vis == "v1" else "V2, " + lab}')
        A = {'OT-FiLM (ours)': load([f'{R}/eval/ep006000_film_n100_s{s}_on_{reg}.json' for s in (1, 2, 3)], vis),
             'EB-VAE (Baaz 2026)': load([f'{R}/baaz/eval_test_{reg}_seed{s}.json' for s in (1, 2, 3)], vis)}
        for name, runs in A.items():
            if not runs: print(f'   {name:20s} n/a'); continue
            m = summary(runs)
            print(f'   {name:20s} med {m["med"][0]:6.1f}  bias {m["bias"][0]:+6.1f}  w20 {m["w20"][0]:5.1f}  trm {m["trm"][0]:6.1f}  n>100% {m["n100"][0]:5.1f}'
                  f'   [n={len(runs[0][0])}, seeds med {" ".join(f"{x:.1f}" for x in m["med"][1])}]')
        a, b = A['OT-FiLM (ours)'], A['EB-VAE (Baaz 2026)']
        if a and b:
            ta, tb = a[0][0], b[0][0]; ea = np.mean([np.abs(p / t - 1) for t, p in a], 0); eb = np.mean([np.abs(p / t - 1) for t, p in b], 0)
            o = np.argsort(tb); pos = np.searchsorted(tb[o], ta).clip(0, len(tb) - 1); idx = o[pos]; ok = np.isclose(tb[idx], ta, rtol=1e-3)
            if ok.mean() > 0.99 and len(set(idx[ok])) == ok.sum():
                d = ea[ok] - eb[idx[ok]]
                print(f'   paired (n={ok.sum()}): ours - theirs median |err| {100 * (np.median(ea[ok]) - np.median(eb[idx[ok]])):+.2f} pt; '
                      f'ours closer for {100 * np.mean(d < 0):.1f}% of patients; Wilcoxon p = {wilcoxon(d).pvalue:.2g}')
            else:
                print(f'   paired comparison skipped: true-AUC matching {100 * ok.mean():.1f}% / not one-to-one')
