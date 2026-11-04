#!/usr/bin/env python3
"""Realistic MAP-BE (SAEM in Monolix on the training patients, then MAP) vs the oracle and the learned models, linear and
saturable (Km 500) simulators, one seed per test (sparse s1, dense s0) -- Per model: error on the observed
regimen, mean over the 5 unseen regimens of the median |AUC 0-30 h err| % (new amounts / new times), dose elasticity
350 -> 850 mg (truth 0.97 linear / 1.26 saturable), and the median SIGNED error of the two dose-extrapolations split by the arm
the test patient was observed on: observed on 800 mg -> predicted at 350 mg, observed on 400 mg -> predicted at 850 mg
(patient route: ~0 / ~0 if the model gets the dose scaling right)."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, json, numpy as np
R = 'results/regimen_dh'; EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'
REGS = ('train', 'amt350', 'amt600', 'amt850', 'time400', 'time800')
CONFIGS = (('sparse', 1), ('dense', 0))


def path(m, sim, d, s):
    mm = 'mm500/' if sim == 'mm500' else ''; tag = f'N100_{d}_s{s}.json'
    return {'oracle': f'{R}/{mm}mapbe_oracle/{tag}',
            'saem_lin': f'{R}/mapbe_pop/{sim}/saem_lin/{tag}', 'saem_mm': f'{R}/mapbe_pop/{sim}/saem_mm/{tag}',
            'true_lin': f'{R}/mapbe_pop/{sim}/true_lin/{tag}',
            'v3': f'{R}/v3/{sim}/lam1000/{tag}', 'B2': f'{R}/{mm}B2/{tag}', 'B': f'{R}/{mm}B/{tag}',
            'ebvae': f'{EB}/res_mm500/ctrl/{tag}' if sim == 'mm500' else f'{EB}/res_titr/ctrl/{tag}'}[m]


MODELS = (('oracle', 'MAP-BE oracle: true model + true population'),
          ('saem_mm', 'MAP-BE saturable model, SAEM population'),
          ('saem_lin', 'MAP-BE LINEAR model, SAEM population'),
          ('true_lin', 'MAP-BE linear model, true (ka,ke,V) population'),
          ('v3', 'improved B (penalty 1000)'), ('B2', 'B2 (linear)'), ('B', 'B'), ('ebvae', 'EB-VAE'))
# truth per test patient for the elasticity: from any file that stores predictions and errors (same test patients everywhere)
def truth_auc(sim, d, s):
    for m in ('saem_lin', 'oracle', 'v3'):
        p = path(m, sim, d, s)
        if os.path.exists(p):
            j = json.load(open(p))
            if 'auc_pred_per_patient' in j['individual']['amt350']:
                return {g: np.array(j['individual'][g]['auc_pred_per_patient']) / (1 + np.array(j['individual'][g]['auc_err_per_patient']))
                        for g in ('amt350', 'amt850')}


k = np.log(850 / 350)
for sim, lab in (('lin', 'LINEAR simulator (true elasticity 0.97)'), ('mm500', 'SATURABLE simulator, Km 500 (true elasticity 1.26)')):
    print(f'\n=== {lab}')
    for d, s in CONFIGS:
        tr = truth_auc(sim, d, s)
        print(f'-- N100 {d} s{s}   {"model":48s} {"obs.":>5s} {"unseen":>6s} ({"amts":>5s}/{"times":>5s}) {"elast.":>6s} | signed err: '
              f'800->350  400->850')
        for m, ml in MODELS:
            p = path(m, sim, d, s)
            if not os.path.exists(p):
                print(f'   {ml:66s} n/a'); continue
            j = json.load(open(p))['individual']
            a = {g: j[g]['auc_abs_median'] for g in REGS}
            if any(not np.isfinite(v) or v > 1e4 for v in a.values()):
                print(f'   {ml:66s} training diverged (errors up to {max(a.values()):.1e} %)'); continue
            un = np.mean([a[g] for g in REGS[1:]]); am = np.mean([a[g] for g in REGS[1:4]]); ti = np.mean([a[g] for g in REGS[4:]])
            e = {g: np.array(j[g]['auc_err_per_patient']) for g in ('amt350', 'amt850')}
            el = np.nan
            if tr is not None:
                with np.errstate(invalid='ignore', divide='ignore'):
                    el = np.nanmedian(np.log((1 + e['amt850']) * tr['amt850'] / ((1 + e['amt350']) * tr['amt350'])) / k)
            print(f'   {ml:66s} {a["train"]:5.1f} {un:6.1f} ({am:5.1f}/{ti:5.1f}) {el:6.2f} |   {100 * np.median(e["amt350"][1::2]):+6.1f}   '
                  f'{100 * np.median(e["amt850"][0::2]):+6.1f}')
print('\nSAEM population estimates (truth: ka 0.4, ke 0.6, V 50, omega 0.5, sd 10; Km 500):')
for sim in ('lin', 'mm500'):
    for d, s in CONFIGS:
        for st in ('lin', 'mm'):
            f = f'{R}/mapbe_pop/saem/{sim}_{d}_s{s}_{st}/estimates.json'
            if os.path.exists(f):
                j = json.load(open(f)); e = j['estimates']
                print(f'   {sim:5s} {d:6s} {st:3s}: ' + ', '.join(f'{n} {v:.3g}' for n, v in e.items()) +
                      f' | -2LL {j["loglik"]["importanceSampling"][0]:.1f}, BIC {j["loglik"]["importanceSampling"][2]:.1f}')
