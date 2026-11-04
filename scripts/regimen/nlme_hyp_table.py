#!/usr/bin/env python3
"""Two hypotheses for the extra exposure of the 800 mg arm as NLME models, fitted by SAEM (Monolix)
on the 200 training patients, then MAP-BE on the 400 test patients (queue_nlme_hyp.sh, queue_mapbe_pop.sh):
  PATIENT route: linear model + arm covariate (log theta = log theta_pop + beta * [arm 800] + eta) on ke, or on ka/ke/V
  DRUG route:    saturable model (shared Km), same population in both arms
  BOTH:          saturable model + arm covariate on ka/ke/V (does the arm effect vanish once saturation is available?)
Fit to the OBSERVED data: -2LL and BIC (importance sampling); the arm effects as % change of the typical value (800 vs 400 mg).
Counterfactual: observed-regimen and unseen-regimen error (mean over the 5 unseen regimens of the median |AUC 0-30 h err| %),
dose elasticity 350 -> 850 mg, median signed error observed-on-800 -> 350 mg and observed-on-400 -> 850 mg."""
import os, json, numpy as np
R = 'results/regimen_dh'; D = f'{R}/mapbe_pop'
REGS = ('train', 'amt350', 'amt600', 'amt850', 'time400', 'time800')
VARIANTS = (('lin', 'linear, same population in both arms'), ('lin_armke', 'PATIENT: linear + arm effect on ke'),
            ('lin_armall', 'PATIENT: linear + arm effect on ka, ke, V'), ('mm', 'DRUG: saturable (shared Km), same population'),
            ('mm_armall', 'BOTH: saturable + arm effect on ka, ke, V'))
k = np.log(850 / 350)
for sim, simlab in (('mm500', 'SATURABLE simulator, Km 500 (true elasticity 1.26; arms randomised: true arm effect 0)'),
                    ('lin', 'LINEAR simulator (negative control: true arm effect 0, no saturation)')):
    print(f'\n=== {simlab}')
    for d, s in (('sparse', 1), ('dense', 0)):
        tag = f'{sim}_{d}_s{s}'
        print(f'-- N100 {d} s{s}\n   {"model":44s} {"npar":>4s} {"-2LL":>9s} {"BIC":>9s} {"dBIC":>7s}  {"arm effect % (ka/ke/V)":>24s} {"Km":>5s} | '
              f'{"obs.":>5s} {"unseen":>6s} {"elast.":>6s} {"800->350":>8s} {"400->850":>8s}')
        rows = []
        for v, lab in VARIANTS:
            f = f'{D}/saem/{tag}_{v}/estimates.json'; o = f'{D}/{sim}/saem_{v}/N100_{d}_s{s}.json'
            if not os.path.exists(f):
                continue
            j = json.load(open(f)); e = j['estimates']; ll = j['loglik']['importanceSampling']
            npar = len(e)
            arm = '/'.join(f'{100 * (np.exp(e[f"beta_{n}_ARM800"]) - 1):+.0f}' if f'beta_{n}_ARM800' in e else '-' for n in ('ka', 'ke', 'V'))
            km = f'{e["Km_pop"]:.0f}' if 'Km_pop' in e else '-'
            cf = ''
            if os.path.exists(o):
                r = json.load(open(o))['individual']
                un = np.mean([r[g]['auc_abs_median'] for g in REGS[1:]])
                p350, p850 = np.array(r['amt350']['auc_pred_per_patient']), np.array(r['amt850']['auc_pred_per_patient'])
                el = np.median(np.log(p850 / p350) / k)
                cf = (f'{r["train"]["auc_abs_median"]:5.1f} {un:6.1f} {el:6.2f} {100 * np.median(np.array(r["amt350"]["auc_err_per_patient"])[1::2]):+8.1f} '
                      f'{100 * np.median(np.array(r["amt850"]["auc_err_per_patient"])[0::2]):+8.1f}')
            rows.append((lab, npar, ll[0], ll[2], arm, km, cf))
        if not rows:
            print('   (not fitted yet)'); continue
        best = min(r[3] for r in rows)
        for lab, npar, m2ll, bic, arm, km, cf in rows:
            print(f'   {lab:44s} {npar:4d} {m2ll:9.1f} {bic:9.1f} {bic - best:+7.1f}  {arm:>24s} {km:>5s} | {cf}')
print('\nnpar = population parameters (fixed effects, arm effects, omegas, correlations, residual sd). dBIC > 10 = very strong evidence '
      'against the model. Arm effect = % change of the typical value in the 800 mg arm.')
