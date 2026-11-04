#!/usr/bin/env python3
"""Prior-gauge test (queue_eb_priortest.sh; ): EB-VAE with its usual dose-conditioned prior (res_tacro_s2/ctrl) vs a
prior that ignores the amount (res_tacro_s2/prior_nodose), tacrolimus scenario 2 (linear), dense, seeds 0-2, + the dose-event model.
Unseen / training-regimen error (median |AUC err| %), and the median SIGNED error per regimen split by the arm the test patient was
observed on (2 mg | 4 mg; even / odd index). Prediction (Prop. priorgauge (iii)): the split shrinks if the prior carries it."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, json, numpy as np
R = 'results/regimen_dh'; EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'
REGS = ('train', 'amt1.75', 'amt3', 'amt4.25', 'timeLOW', 'timeHIGH')
for lab, base in (('EB-VAE, prior reads the dose (ctrl)', f'{EB}/res_tacro_s2/ctrl'), ('EB-VAE, prior IGNORES the dose', f'{EB}/res_tacro_s2/prior_nodose'),
                  ('dose-event (ours)', f'{R}/tacro_s2/Bfixkp3.1n')):
    js = [json.load(open(f'{base}/N100_dense_s{s}.json')) for s in range(3) if os.path.exists(f'{base}/N100_dense_s{s}.json')]
    if not js:
        print(f'{lab}: n/a'); continue
    un = [np.mean([j['individual'][g]['auc_abs_median'] for g in REGS[1:]]) for j in js]; tr = [j['individual']['train']['auc_abs_median'] for j in js]
    print(f'{lab} (n {len(js)}): unseen {np.mean(un):.1f} +- {np.std(un, ddof=1) if len(un) > 1 else 0:.1f} {np.round(un, 1).tolist()}, training {np.mean(tr):.1f}')
    sp = {g: [(np.median(100 * np.array(j['individual'][g]['auc_err_per_patient'])[0::2]),
               np.median(100 * np.array(j['individual'][g]['auc_err_per_patient'])[1::2])) for j in js] for g in REGS}
    print('   signed error, observed on 2 mg | 4 mg:  ' + '  '.join(f'{g} {np.mean([x[0] for x in v]):+5.1f}|{np.mean([x[1] for x in v]):+5.1f}' for g, v in sp.items()))
