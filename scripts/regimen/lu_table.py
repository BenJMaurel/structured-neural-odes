#!/usr/bin/env python3
"""Lu comparison on Baaz's regimen test (queue_lu.sh). N100 sparse seeds 0-2 + N100 dense seed 0; control =
alternating training arms, titrated = gen_regimens.titrate(rho 0.8: fast eliminators get 800 mg); test arms always alternating.
Models: Lu iScience as published (encoder fed the regimen being predicted) and the same trained model with the factual
encoder, Lu NMI, B, B2, EB-VAE (titrated copy; control rerun through the copy, the original run_regimen.py JSON as fallback).
  1. unseen-regimen error: mean over the 5 unseen regimens of the median |AUC(0-30 h) err| %; titrated minus control (DiD).
  2. dose elasticity d log AUC / d log dose between 350 and 850 mg (4 doses at 0/8/16/24 h), per patient, vs the truth
     (< 1 because the central amount starts at v whatever the dose): median predicted vs true, median signed error.
  3. test 1 (iScience): identity ratio of the same patient's z0 when only the regimen being predicted changes (0 = the patient
     does not move, 1 = as far as two random patients), and transfer of a log ke probe fitted on z0(train regimen)."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, warnings, numpy as np
warnings.filterwarnings('ignore', 'invalid value encountered in log')
sys.path.insert(0, f'{REPO}/third_party/baaz_ebvae/ours/regimen')
import gen_regimens as G
R, RHO = 'results/regimen_dh', 0.8
EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'
CONFIGS = (('sparse', (0, 1, 2)), ('dense', (0,)))
REGS = ('train', 'amt350', 'amt600', 'amt850', 'time400', 'time800')
EG = np.arange(0, 30.0001, 0.25)
MODELS = (('luis', 'Lu iScience, as published (target regimen in the encoder)'),
          ('luis_fact', 'Lu iScience, factual encoder (two-port fix)'),
          ('lunmi', 'Lu NMI (two-port, frozen parameters)'),
          ('ebvae', 'EB-VAE (dose inside the vector field)'),
          ('B', 'B (dose = kick, physiology conserved)'),
          ('B2', 'B2 (B + linear compartments)'))


def load(model, cond, d, s):
    tag = f'N100_{d}_s{s}'
    base = 'luis' if model == 'luis_fact' else model
    if base == 'ebvae':
        p = f'{EB}/res_titr/{"ctrl" if cond == "ctrl" else f"titr{RHO}"}/{tag}.json'
        if cond == 'ctrl' and not os.path.exists(p):
            p = f'{EB}/res/N100_{d}' + (f'_s{s}' if s else '') + '.json'
    else:
        p = f'{R}/{base}/{tag}.json' if cond == 'ctrl' else f'{R}/titr{RHO}/{base}/{tag}.json'
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    if model == 'luis_fact':
        j = dict(j, individual=j['individual_factual_encoder'])
    return j


_truth = {}
def true_auc(d, s):
    """test-patient AUC(0-30 h) per regimen, regenerated with run_regimen.py's random sequence (N100)"""
    if (d, s) not in _truth:
        SAMP = np.arange(0, 30.25, 0.5) if d == 'dense' else np.array([0, 2, 7.5, 15.5, 23.5, 28.0])
        rng = np.random.default_rng(s); P_tr = G.draw(200, rng)
        A_tr = np.repeat(np.where(np.arange(200) % 2 == 0, 400.0, 800.0)[:, None], 4, 1)
        te, c = G.simulate(P_tr, [0, 8, 16, 24], A_tr); G.noisy(np.stack([np.interp(SAMP, te, ci) for ci in c]), rng)
        P_te = G.draw(400, rng); out = {}
        for name in ('amt350', 'amt850'):
            times, amts = G.REGIMENS[name]
            te, c = G.simulate(P_te, times, np.tile(np.array(amts, float), (400, 1)))
            out[name] = np.trapezoid(np.stack([np.interp(EG, te, ci) for ci in c]), EG, axis=1)
        _truth[(d, s)] = out
    return _truth[(d, s)]


def pred_auc(j, name, d, s):
    ind = j['individual'][name]
    if 'auc_pred_per_patient' in ind:
        p = np.array(ind['auc_pred_per_patient'])
        if 'auc_err_per_patient' in ind:   # the regenerated truth must be the one the run used
            t = p / (1 + np.array(ind['auc_err_per_patient']))
            assert np.allclose(t, true_auc(d, s)[name], rtol=1e-4), f'regenerated truth differs ({name}, {d}, s{s})'
        return p
    if 'auc_err_per_patient' in ind:
        return (1 + np.array(ind['auc_err_per_patient'])) * true_auc(d, s)[name]
    return None


def elasticity(j, d, s):
    a350, a850 = pred_auc(j, 'amt350', d, s), pred_auc(j, 'amt850', d, s)
    if a350 is None:
        return None
    t = true_auc(d, s); k = np.log(850 / 350)
    ep, et = np.log(a850 / a350) / k, np.log(t['amt850'] / t['amt350']) / k
    return ep, et


unseen = lambda j: np.mean([j['individual'][g]['auc_abs_median'] for g in REGS[1:]])
print('1. Unseen-regimen error: mean over the 5 unseen regimens of the median |AUC(0-30 h) err| % (factual regimen in brackets)')
print('   control = alternating training arms, titrated = fast eliminators on 800 mg (rho 0.8); DiD = titrated - control')
for d, seeds in CONFIGS:
    print(f'-- N100 {d}, seeds {list(seeds)}')
    for m, lab in MODELS:
        cells = {}
        for cond in ('ctrl', 'titr'):
            js = [j for s in seeds for j in [load(m, cond, d, s)] if j is not None]
            cells[cond] = (np.mean([unseen(j) for j in js]), np.mean([j['individual']['train']['auc_abs_median'] for j in js]),
                           [round(float(unseen(j)), 1) for j in js]) if js else None
        c, t = cells['ctrl'], cells['titr']
        fmt = lambda x: f'{x[0]:6.1f} ({x[1]:4.1f})' if x else '    n/a     '
        did = f'{t[0] - c[0]:+6.1f}' if c and t else '   n/a'
        print(f'   {lab:58s} control {fmt(c)}  titrated {fmt(t)}  DiD {did}   seeds ctrl {c[2] if c else "-"} titr {t[2] if t else "-"}')

print('\n2. Dose elasticity d log AUC / d log dose (350 -> 850 mg), test patients: median predicted [true]; median signed error')
for d, seeds in CONFIGS:
    print(f'-- N100 {d}')
    for m, lab in MODELS:
        row = []
        for cond in ('ctrl', 'titr'):
            es = [e for s in seeds for j in [load(m, cond, d, s)] if j is not None for e in [elasticity(j, d, s)] if e is not None]
            if es:
                ep = np.concatenate([e[0] for e in es]); et = np.concatenate([e[1] for e in es])
                ok = np.isfinite(ep)   # a predicted AUC <= 0 (models without a positivity constraint) has no log
                row.append(f'{cond} {np.median(ep[ok]):5.2f} [{np.median(et[ok]):4.2f}] err {np.median(ep[ok] - et[ok]):+5.2f}'
                           + (f' ({(~ok).sum()} invalid)' if (~ok).any() else ''))
            else:
                row.append(f'{cond}   n/a')
        print(f'   {lab:58s} ' + '   '.join(row))

print('\n3. Test 1 (Lu iScience): the same patient and observations, encoder fed each regimen to be predicted')
for d, seeds in CONFIGS:
    for cond in ('ctrl', 'titr'):
        js = [j for s in seeds for j in [load('luis', cond, d, s)] if j is not None]
        if not js:
            continue
        idr = ' '.join(f'{g} {np.mean([j["identity"][g] for j in js]):.2f}' for g in REGS[1:])
        pr = ' '.join(f'{g} {np.mean([j["identity_probe"]["log ke"][g] for j in js]):+.2f}' for g in REGS)
        print(f'   N100 {d:6s} {cond:4s} identity ratio vs z0(train): {idr}')
        print(f'   {"":17s} log ke probe R^2 (fitted on z0(train), CV {np.mean([j["identity_probe"]["log ke"]["cv_r2_train"] for j in js]):.2f}): {pr}')
print('   B, B2, Lu NMI, EB-VAE: the encoder never sees the regimen to be predicted -> identity ratio 0 by construction.')
