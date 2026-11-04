#!/usr/bin/env python3
"""Improved B with the TRUE observation noise (10 mg/L) vs the default fixed noise (0.05*YMAX = 81 / 73 mg/L) on the saturable
simulator (Km 500, true elasticity 1.26), with the NLME references. One seed per test.
Columns: validation RMSE (fresh patients, training regimen; mg/L), observed-regimen error, mean over the 5 unseen regimens of the
median |AUC 0-30 h err| % (new amounts / new times), dose elasticity 350 -> 850 mg, median signed error observed-on-800 -> 350 mg
and observed-on-400 -> 850 mg (0 / 0 = dose scaling right; +28 / -20 = the linear-model pattern), correction size dyn/read."""
import os, json, numpy as np
R = 'results/regimen_dh'; REGS = ('train', 'amt350', 'amt600', 'amt850', 'time400', 'time800'); k = np.log(850 / 350)


def truth(d, s):
    j = json.load(open(f'{R}/mapbe_pop/mm500/saem_lin/N100_{d}_s{s}.json'))['individual']
    return {g: np.array(j[g]['auc_pred_per_patient']) / (1 + np.array(j[g]['auc_err_per_patient'])) for g in ('amt350', 'amt850')}


def row(lab, p, tr):
    if not os.path.exists(p):
        print(f'   {lab:48s} (not done)'); return
    j = json.load(open(p)); r = j['individual']
    un = np.mean([r[g]['auc_abs_median'] for g in REGS[1:]]); am = np.mean([r[g]['auc_abs_median'] for g in REGS[1:4]])
    ti = np.mean([r[g]['auc_abs_median'] for g in REGS[4:]])
    e = {g: np.array(r[g]['auc_err_per_patient']) for g in ('amt350', 'amt850')}
    with np.errstate(invalid='ignore', divide='ignore'):
        el = np.nanmedian(np.log((1 + e['amt850']) * tr['amt850'] / ((1 + e['amt350']) * tr['amt350'])) / k)
    val = j.get('validation', {}).get('rmse_mgL', np.nan)
    cr = (f'{j["dyn_correction_rel"]:.3f}/{j["read_correction_rel"]:.3f}' if 'dyn_correction_rel' in j else '')
    print(f'   {lab:48s} {val:6.1f} | {r["train"]["auc_abs_median"]:5.1f} {un:6.1f} ({am:5.1f}/{ti:5.1f}) {el:6.2f} | '
          f'{100 * np.median(e["amt350"][1::2]):+7.1f} {100 * np.median(e["amt850"][0::2]):+7.1f} | {cr}')


for d, s in (('sparse', 1), ('dense', 0)):
    tr = truth(d, s); tag = f'N100_{d}_s{s}.json'
    print(f'\n=== SATURABLE simulator, N100 {d} s{s} (true elasticity 1.26)')
    print(f'   {"model":48s} {"valid.":>6s} | {"obs.":>5s} {"unseen":>6s} ({"amts":>5s}/{"times":>5s}) {"elast.":>6s} | '
          f'{"800->350":>7s} {"400->850":>7s} | corr dyn/read')
    row('MAP-BE oracle (true model + population)', f'{R}/mm500/mapbe_oracle/{tag}', tr)
    row('MAP-BE saturable, SAEM population', f'{R}/mapbe_pop/mm500/saem_mm/{tag}', tr)
    row('MAP-BE linear, SAEM population', f'{R}/mapbe_pop/mm500/saem_lin/{tag}', tr)
    print('   -- default fixed noise 0.05*YMAX (81 / 73 mg/L)')
    row('B2', f'{R}/mm500/B2/{tag}', tr)
    for lam in ('0', '100', '1000', '10000'):
        row(f'improved B, penalty {lam}', f'{R}/v3/mm500/lam{lam}/{tag}', tr)
    print('   -- TRUE noise 10 mg/L')
    row('B2', f'{R}/v3/mm500_truenoise/B2/{tag}', tr)
    for lam in ('0', '1000', '10000', '100000'):
        row(f'improved B, penalty {lam}', f'{R}/v3/mm500_truenoise/lam{lam}/{tag}', tr)
print('\nvalid. = RMSE on 100 fresh patients under the training regimen (true noise sd 10 mg/L). The penalty multiplies mean(corr^2); at '
      '10 mg/L the data term is ~64x stronger than at 81 mg/L, so penalty 1e5 here ~ 1600 there.')
