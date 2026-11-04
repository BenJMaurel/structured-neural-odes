#!/usr/bin/env python3
"""EB-VAE at UNSEEN DOSING TIMES: table for third_party/baaz_ebvae/ours/regimen/res_timing/
(run_regimen_timing.py, queue scripts/regimen/queue_ebtiming.sh).
  1. reproduction: the 6 original regimens must equal res_titr/ctrl/<run>.json (same training) -- asserted;
  2. median |AUC(0-30 h) err| and median signed error per regimen, EB-VAE vs MAP-BE oracle (true model, true population,
     true noise; MAP on the same factual samples, as scripts/regimen/mapbe_regimen.py -- its 6 original regimens must
     reproduce results/regimen_dh/mapbe_oracle/<run>.json, asserted);
  3. where the error sits: median signed AUC error per window between consecutive doses (and the tail after the last);
  4. kick of a later dose: extra AUC over [t_dose, 30 h] relative to single800, model / truth (median, IQR).
    ebtiming_table.py  -> results/regimen_dh/ebtiming_table.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, glob
import numpy as np, torch
os.chdir(REPO)
RD = f'{REPO}/third_party/baaz_ebvae/ours/regimen'; sys.path.insert(0, RD)
import gen_regimens as G
assert G.KM is None, 'linear simulator only'
src = open(f'{RD}/run_regimen_timing.py').read()
NEW = eval(src.split('NEW_REGIMENS = ')[1].split('\n}\n')[0] + '\n}')
REGS = {**G.REGIMENS, **NEW}
torch.set_default_dtype(torch.float64)
DT = G.DT; t_eval = np.arange(0.0, G.T_END + DT / 2, DT)
mu = torch.tensor(np.log([G.KA, G.KE, G.V])); Om_inv = torch.linalg.inv(torch.tensor(np.outer([G.SD] * 3, [G.SD] * 3) * G.CORR))
SIG = float(G.ADD_E)


def conc_at(eta, times, amounts, idx):   # closed form of gen_regimens.simulate (copied from mapbe_regimen.py; checked below)
    ka, ke, v = torch.exp(eta).unbind(-1)
    ke = torch.where((ka - ke).abs() < 1e-7, ke + 1e-7, ke)
    a, gg = 1 - ke * DT, 1 - ka * DT
    n = torch.tensor(idx, dtype=torch.float64); c = v[:, None] * a[:, None] ** n[None, :]
    for kd, k in enumerate([int(np.argmax(t_eval >= t)) for t in times]):
        m = (n - k).clamp(min=0)[None, :]
        c = c + (n[None, :] >= k) * torch.tensor(amounts[:, kd])[:, None] * ka[:, None] * (a[:, None] ** m - gg[:, None] ** m) / (ka - ke)[:, None]
    return c


def oracle(P_te, arm, y_te, SAMP):
    obs_idx = np.array([int(np.argmin(np.abs(t_eval - s))) for s in SAMP]); A0 = np.repeat(arm[:, None], 4, 1)
    c_cf = conc_at(torch.tensor(np.log(P_te[:20])), [0, 8, 16, 24], A0[:20], np.arange(len(t_eval))).numpy()
    _, c_sim = G.simulate(P_te[:20], [0, 8, 16, 24], A0[:20]); assert np.abs(c_cf - c_sim).max() / np.abs(c_sim).max() < 1e-9
    Y = torch.tensor(y_te); eta = mu.repeat(len(Y), 1).clone().requires_grad_(True)
    opt = torch.optim.LBFGS([eta], lr=1, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn='strong_wolfe')
    def objective():
        opt.zero_grad(); c = conc_at(eta, [0, 8, 16, 24], A0, obs_idx); d = eta - mu
        f = ((Y - c) ** 2).sum() / (2 * SIG ** 2) + 0.5 * torch.einsum('bi,ij,bj->', d, Om_inv, d); f.backward(); return f
    for _ in range(5):
        opt.step(objective)
    return torch.exp(eta.detach()).numpy()


def amounts(name, arm):
    times, amts = REGS[name]
    return times, (np.repeat(arm[:, None], len(times), 1) if amts is None else np.tile(np.array(amts, float), (len(arm), 1)))


def auc(c, EG, a=0.0, b=30.0):
    m = (EG >= a - 1e-9) & (EG <= b + 1e-9); return np.trapezoid(c[:, m], EG[m], axis=1)


out = []; P = lambda s='': out.append(s)
P('EB-VAE (Baaz et al. public code) at UNSEEN DOSING TIMES -- training doses always at 0, 8, 16, 24 h (400 or 800 mg, randomised arms).')
P('Cells: median |AUC(0-30 h) err| % (median signed err %); oracle = MAP-BE with the true model + true population on the same samples.')
P('New regimens: ' + '; '.join(f'{k} {v[0]} h x {v[1][0]:g} mg' for k, v in NEW.items()))
runs = sorted(glob.glob(f'{RD}/res_timing/N100_*.json'))
for fj in runs:
    tag = os.path.basename(fj)[:-5]; r = json.load(open(fj)); d = np.load(fj[:-5] + '_curves.npz'); EG = d['EG']
    ref = json.load(open(f'{RD}/res_titr/ctrl/{tag}.json'))
    for k in G.REGIMENS:   # 1. reproduction of the earlier run (identical training)
        for q in ('auc_abs_median', 'auc_bias_median'):
            assert abs(r['individual'][k][q] - ref['individual'][k][q]) < 1e-9, (tag, k, q, r['individual'][k][q], ref['individual'][k][q])
    th = oracle(d['P_te'], d['arm_te'], d['y_te'], d['SAMP'])
    orc = {}
    for k in REGS:
        times, A = amounts(k, d['arm_te']); _, c = G.simulate(th, times, A)
        pr = np.stack([np.interp(EG, t_eval, ci) for ci in c]); e = auc(pr, EG) / auc(d['truth_' + k], EG) - 1
        orc[k] = (100 * np.median(np.abs(e)), 100 * np.median(e))
    oref = json.load(open(f'results/regimen_dh/mapbe_oracle/{tag}.json'))
    for k in G.REGIMENS:
        assert abs(orc[k][0] - oref['individual'][k]['auc_abs_median']) < 0.05, (tag, k, orc[k][0], oref['individual'][k]['auc_abs_median'])
    P(); P(f'=== {tag}   (original regimens reproduce res_titr/ctrl: OK; oracle reproduces mapbe_oracle: OK; '
           f'grid check max {max(r["grid_check"].values()):.3f} %)')
    pk_tr = np.median(d['truth_train'][d['arm_te'] == 800].max(1))
    P(f'  {"regimen":10s} {"EB-VAE":>16s} {"oracle":>14s}   true peak / training-800 peak (median)')
    for k in REGS:
        i = r['individual'][k]
        P(f'  {k:10s} {i["auc_abs_median"]:6.1f} ({i["auc_bias_median"]:+6.1f})  {orc[k][0]:5.1f} ({orc[k][1]:+5.1f})   '
          f'{np.median(d["truth_" + k].max(1)) / pk_tr:4.2f}' + ('' if k in G.REGIMENS else '   NEW'))
    P('  where the error sits -- median signed AUC err % per window [dose j, dose j+1) and the tail [last dose, 30 h]:')
    for k in ['train', 'time800'] + list(NEW):
        times = sorted(REGS[k][0]); edges = (times if times[0] == 0 else [0] + times) + [30]
        pred, tru = d[k], d['truth_' + k]
        if k == 'train':
            pred, tru = pred[d['arm_te'] == 800], tru[d['arm_te'] == 800]
        cells = [f'{a:g}-{b:g}h {100 * np.median(auc(pred, EG, a, b) / auc(tru, EG, a, b) - 1):+6.1f}' for a, b in zip(edges[:-1], edges[1:])]
        P(f'    {k + (" (800 arm)" if k == "train" else ""):18s} ' + '  '.join(cells))
    P('  kick of later doses vs single800: extra AUC over [t, 30 h], model / truth, median (IQR):')
    for k, t in [('gap800', 20), ('q12h800', 12), ('burst800', 0)]:
        ex_p = auc(d[k], EG, t) - auc(d['single800'], EG, t); ex_t = auc(d['truth_' + k], EG, t) - auc(d['truth_single800'], EG, t)
        q = np.percentile(ex_p / ex_t, [25, 50, 75])
        P(f'    {k:9s} doses after 0 h, extra AUC over [{t:g}, 30 h]: {q[1]:.2f} ({q[0]:.2f}-{q[2]:.2f})')
txt = '\n'.join(out); print(txt)
open('results/regimen_dh/ebtiming_table.txt', 'w').write(txt + '\n')
