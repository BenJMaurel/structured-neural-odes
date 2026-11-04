#!/usr/bin/env python3
"""Improved B (dh_regimen_v3.py) vs baselines on Baaz's regimen test, LINEAR and SATURABLE (Km 500) simulators.
Per simulator x design (sparse seeds 0-2, dense seed 0): validation RMSE (fresh patients, the penalty-selection criterion), error on
the observed regimen, mean error over the 5 unseen regimens (new amounts / new times), dose elasticity 350 -> 850 mg (predicted
median, truth in brackets), for randomised and titrated training; size of the learned corrections relative to the linear part."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, glob, numpy as np
sys.path.insert(0, f'{REPO}/third_party/baaz_ebvae/ours/regimen')
import gen_regimens as G
R = 'results/regimen_dh'; EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'; KM = 500.0
REGS = ('train', 'amt350', 'amt600', 'amt850', 'time400', 'time800'); EG = np.arange(0, 30.0001, 0.25)
CONFIGS = (('sparse', (1,)), ('dense', (0,)))   # one seed per test; every model on the same seed
SEL = json.load(open(f'{R}/v3/selected_lambda.json')) if os.path.exists(f'{R}/v3/selected_lambda.json') else {}


def load(p):
    return json.load(open(p)) if p and os.path.exists(p) else None


def path(model, sim, cond, d, s):
    mm = sim != 'lin'; tag = f'N100_{d}_s{s}.json'
    if model == 'mapbe':
        return f'{R}/{"mm500/" if mm else ""}mapbe_oracle/{tag}' if cond == 'ctrl' else None
    if model == 'ebvae':
        base = f'{EB}/res_mm500' if mm else f'{EB}/res_titr'
        p = f'{base}/{"ctrl" if cond == "ctrl" else "titr0.8"}/{tag}'
        if not mm and cond == 'ctrl' and not os.path.exists(p):
            p = f'{EB}/res/N100_{d}' + (f'_s{s}' if s else '') + '.json'
        return p
    if model in ('B', 'B2'):
        return f'{R}/{"mm500/" if mm else ""}{"titr0.8/" if cond == "titr" else ""}{model}/{tag}'
    if model == 'noise':
        return f'{R}/v3/{sim}/noise_learned_nocorr/{tag}' if cond == 'ctrl' else None
    if model.startswith('lam'):
        return f'{R}/v3/{sim}/{"titr0.8/" if cond == "titr" else ""}{model}/{tag}'


_truth = {}
def true_auc(d, s, sim):
    key = (d, s, sim)
    if key not in _truth:
        old = G.KM; G.KM = None if sim == 'lin' else KM
        SAMP = np.arange(0, 30.25, 0.5) if d == 'dense' else np.array([0, 2, 7.5, 15.5, 23.5, 28.0])
        rng = np.random.default_rng(s); P_tr = G.draw(200, rng)
        A_tr = np.repeat(np.where(np.arange(200) % 2 == 0, 400.0, 800.0)[:, None], 4, 1)
        te, c = G.simulate(P_tr, [0, 8, 16, 24], A_tr); G.noisy(np.stack([np.interp(SAMP, te, ci) for ci in c]), rng)
        P_te = G.draw(400, rng); out = {}
        for n in ('amt350', 'amt850'):
            times, amts = G.REGIMENS[n]; te, c = G.simulate(P_te, times, np.tile(np.array(amts, float), (400, 1)))
            out[n] = np.trapezoid(np.stack([np.interp(EG, te, ci) for ci in c]), EG, axis=1)
        G.KM = old; _truth[key] = out
    return _truth[key]


def elasticity(j, d, s, sim):
    a = {}
    for n in ('amt350', 'amt850'):
        ind = j['individual'][n]
        a[n] = np.array(ind['auc_pred_per_patient']) if 'auc_pred_per_patient' in ind else (1 + np.array(ind['auc_err_per_patient'])) * true_auc(d, s, sim)[n]
    t = true_auc(d, s, sim); k = np.log(850 / 350)
    with np.errstate(invalid='ignore', divide='ignore'):
        ep = np.log(a['amt850'] / a['amt350']) / k
    return np.nanmedian(ep), np.median(np.log(t['amt850'] / t['amt350']) / k)


def cell(model, sim, cond, d, seeds):
    js = [(s, j) for s in seeds for j in [load(path(model, sim, cond, d, s))] if j is not None]
    if not js:
        return None
    U = lambda j: np.mean([j['individual'][g]['auc_abs_median'] for g in REGS[1:]])
    out = dict(n=len(js), fact=np.mean([j['individual']['train']['auc_abs_median'] for _, j in js]), unseen=np.mean([U(j) for _, j in js]),
               amts=np.mean([np.mean([j['individual'][g]['auc_abs_median'] for g in REGS[1:4]]) for _, j in js]),
               times=np.mean([np.mean([j['individual'][g]['auc_abs_median'] for g in REGS[4:]]) for _, j in js]),
               seeds=[round(float(U(j)), 1) for _, j in js])
    el = [elasticity(j, d, s, sim) for s, j in js]
    out['el'], out['el_true'] = np.mean([e[0] for e in el]), np.mean([e[1] for e in el]); out['el_seeds'] = [round(float(e[0]), 2) for e in el]
    v = [j['validation']['rmse_mgL'] for _, j in js if 'validation' in j]
    out['val'] = np.mean(v) if v else np.nan
    cr = [(j.get('dyn_correction_rel', np.nan), j.get('read_correction_rel', np.nan)) for _, j in js]
    cr = np.array(cr, float)
    out['corr'] = np.array([np.nanmean(cr[:, k]) if np.isfinite(cr[:, k]).any() else np.nan for k in (0, 1)])
    return out


lams = sorted({os.path.basename(p) for p in glob.glob(f'{R}/v3/*/lam*')}, key=lambda x: float(x[3:]))
MODELS = [('mapbe', 'MAP-BE, true model (ceiling)'), ('ebvae', 'EB-VAE'), ('B', 'B (nonlinear, learned noise)'), ('B2', 'B2 (linear)'),
          ('noise', 'B2 + learned noise (no corrections)')] + [(l, f'improved B, penalty {l[3:]}') for l in lams]
for sim, simlab in (('lin', 'LINEAR simulator'), ('mm500', 'SATURABLE simulator (Km 500)')):
    print(f'\n=== {simlab}')
    for d, seeds in CONFIGS:
        sel = SEL.get(f'{sim}_{d}')
        print(f'-- N100 {d}, seeds {list(seeds)}; penalty chosen on validation: {sel}')
        print(f'   {"model":38s} {"valid.":>6s} | {"obs.":>5s} {"unseen":>6s} ({"amts":>5s}/{"times":>5s}) {"elast.":>11s} | titrated: {"unseen":>6s} {"elast.":>6s} | corr dyn/read')
        for m, lab in MODELS:
            c, t = cell(m, sim, 'ctrl', d, seeds), cell(m, sim, 'titr', d, seeds)
            if c is None and t is None:
                continue
            mark = ' *' if m == f'lam{sel}' else ''
            a = (f'{c["val"]:6.1f} | {c["fact"]:5.1f} {c["unseen"]:6.1f} ({c["amts"]:5.1f}/{c["times"]:5.1f}) {c["el"]:5.2f} [{c["el_true"]:.2f}]'
                 if c else f'{"":6s} | {"n/a":>5s}{"":30s}')
            b = f'{t["unseen"]:6.1f} {t["el"]:6.2f}' if t else f'{"-":>6s} {"-":>6s}'
            es = f'elast. per seed {c["el_seeds"]}' + (f' -> {t["el_seeds"]}' if t else '') if c else ''
            cr = f'{c["corr"][0]:.2f}/{c["corr"][1]:.2f}' if c is not None and not np.all(np.isnan(c['corr'])) else ''
            print(f'   {lab + mark:38s} {a} | titrated: {b} | {cr}   seeds {c["seeds"] if c else "-"}; {es}')
print('\n* = penalty chosen on the validation cohort (lowest mean validation RMSE); titrated runs only for it and for penalty 0.')
