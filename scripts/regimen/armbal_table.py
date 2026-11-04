#!/usr/bin/env python3
"""Physiology arm-balance test (queue_armbal.sh; ): improved B, saturable simulator (Km 500, true elasticity 1.26),
default fixed noise, correction penalty 1000, balance weight w on the encoded physiology p only. One seed per test.
Columns: validation RMSE (mg/L), observed-regimen error, mean unseen-regimen error (median |AUC 0-30 h err| %, amounts / times),
dose elasticity 350 -> 850 mg, median signed error observed-on-800 -> 350 mg / observed-on-400 -> 850 mg (linear-model pattern
+28 / -20; right dose scaling 0 / 0), correction size dyn/read, arm difference of p (sum of squared standardised mean
differences; chance ~0.08 train / ~0.04 test). w 0 = the earlier run (p difference from its re-evaluation, reeval/)."""
import os, json, numpy as np
R = 'results/regimen_dh'; REGS = ('train', 'amt350', 'amt600', 'amt850', 'time400', 'time800'); k = np.log(850 / 350)
B = f'{R}/v3/mm500_armbal_default_lam1000'


def truth(d, s):
    j = json.load(open(f'{R}/mapbe_pop/mm500/saem_lin/N100_{d}_s{s}.json'))['individual']
    return {g: np.array(j[g]['auc_pred_per_patient']) / (1 + np.array(j[g]['auc_err_per_patient'])) for g in ('amt350', 'amt850')}


def row(lab, p, tr, pdiff=None):
    if not os.path.exists(p):
        print(f'   {lab:44s} (not done)'); return
    j = json.load(open(p)); r = j['individual']
    un = np.mean([r[g]['auc_abs_median'] for g in REGS[1:]]); am = np.mean([r[g]['auc_abs_median'] for g in REGS[1:4]])
    ti = np.mean([r[g]['auc_abs_median'] for g in REGS[4:]])
    e = {g: np.array(r[g]['auc_err_per_patient']) for g in ('amt350', 'amt850')}
    with np.errstate(invalid='ignore', divide='ignore'):
        el = np.nanmedian(np.log((1 + e['amt850']) * tr['amt850'] / ((1 + e['amt350']) * tr['amt350'])) / k)
    val = j.get('validation', {}).get('rmse_mgL', np.nan)
    cr = f'{j["dyn_correction_rel"]:.3f}/{j["read_correction_rel"]:.3f}' if 'dyn_correction_rel' in j else '-'
    pd = j.get('p_arm_difference') or (json.load(open(pdiff)).get('p_arm_difference') if pdiff and os.path.exists(pdiff) else None)
    pds = f'{pd["train_mean_sq_d"]:6.3f} / {pd["test_mean_sq_d"]:6.3f}' if pd else '-'
    print(f'   {lab:44s} {val:6.1f} | {r["train"]["auc_abs_median"]:5.1f} {un:6.1f} ({am:5.1f}/{ti:5.1f}) {el:6.2f} | '
          f'{100 * np.median(e["amt350"][1::2]):+7.1f} {100 * np.median(e["amt850"][0::2]):+7.1f} | {cr:11s} | {pds}')


for d, s in (('sparse', 1), ('dense', 0)):
    tr = truth(d, s); tag = f'N100_{d}_s{s}.json'
    print(f'\n=== SATURABLE simulator, N100 {d} s{s}, randomised arms')
    print(f'   {"model":44s} {"valid.":>6s} | {"obs.":>5s} {"unseen":>6s} ({"amts":>5s}/{"times":>5s}) {"elast.":>6s} | '
          f'{"800->350":>7s} {"400->850":>7s} | corr dyn/read | p arm diff train / test')
    row('MAP-BE saturable, SAEM population', f'{R}/mapbe_pop/mm500/saem_mm/{tag}', tr)
    row('MAP-BE linear, SAEM population', f'{R}/mapbe_pop/mm500/saem_lin/{tag}', tr)
    row('improved B, no balance (w 0)', f'{R}/v3/mm500/lam1000/{tag}', tr, f'{R}/v3/mm500/lam1000/reeval/{tag}')
    for w in ('0.1', '1', '10'):
        row(f'improved B, balance w {w}', f'{B}/w{w}/{tag}', tr)
    if d == 'sparse':
        print(f'   -- TITRATED training arms (rho 0.8: fast eliminators on 800 mg), balance applied naively')
        row('improved B titrated, no balance (earlier run)', f'{R}/v3/mm500/titr0.8/lam1000/{tag}', tr)
        row('improved B titrated, no balance (rerun, w 0)', f'{B}/titr0.8/w0/{tag}', tr)
        row('improved B titrated, balance w 1', f'{B}/titr0.8/w1/{tag}', tr)
