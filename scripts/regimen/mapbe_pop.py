#!/usr/bin/env python3
"""MAP-BE with a CHOSEN structural model and CHOSEN population parameters on Baaz et al.'s regimen test.
    mapbe_pop.py export <dense|sparse> <seed> <train.csv>
        the 200 training patients (same draws as every learned model) in Monolix format; the data simulator is set by the
        REGIMEN_KM env (unset = linear), exactly as for the other scripts
    mapbe_pop.py fit <dense|sparse> <seed> <out.json> --struct lin|mm --pop true|<monolix estimates.json>
        --struct: the clinician's structural model (lin = 1-cpt first-order absorption; mm = elimination ke*Km*A/(Km + A))
        --pop true: the simulator's population (ka, ke, V, Omega, additive sd 10; Km = the true Km for mm)
        --pop <estimates.json>: SAEM estimates from scripts/monolix/fit_regimen.R on the training patients (realistic MAP-BE)
The MAP step and the scoring are those of mapbe_regimen.py: same test patients, same samples, likelihood on their explicit-Euler
grid. Prediction under each regimen uses the CHOSEN structural model with the patient's MAP parameters (not the true simulator).
Also stored: per-patient predicted AUC, signed error by the arm the test patient was observed on (even index 400 mg, odd 800 mg)."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, runpy, time, argparse
import numpy as np, torch
ROOT = REPO
os.chdir(ROOT); sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['export', 'fit']); ap.add_argument('design', choices=['dense', 'sparse'])
ap.add_argument('seed', type=int); ap.add_argument('out')
ap.add_argument('--struct', choices=['lin', 'mm'], default='lin'); ap.add_argument('--pop', default='true')
ap.add_argument('--n', type=int, default=100)
ap.add_argument('--titrate', type=float, default=None,   # 22/10 opt-in (review): titrated training arms, as dh_regimen*.py --titrate
                help='titrated training arms (gen_regimens.titrate, rho); test patients keep the alternating arms')
ap.add_argument('--arm-col', action='store_true', help='export: add ARM800 (1 = training arm 800 mg) for an arm covariate')
cli = ap.parse_args()
t0 = time.time()
sys.argv = ['x', str(cli.n), cli.design, str(cli.seed), '/tmp/unused.json', '--no-check'] + (['--titrate', str(cli.titrate)] if cli.titrate else [])
g = runpy.run_path('scripts/regimen/dh_regimen_v2.py', run_name='mapbe')      # data only (no training happens)
G, regs, truth, t_eval, EG, SAMP = g['G'], g['regs'], g['truth'], g['t_eval'], g['EG'], g['SAMP']
y_te, A_te, P_te, TT = g['y_te'], g['A_te'], g['P_te'], g['TRAIN_TIMES']
KM_TRUE = G.KM

if cli.mode == 'export':
    y_tr, A_tr = g['y_tr'], g['A_tr']
    lines = ['ID,TIME,AMT,DV' + (',ARM800' if cli.arm_col else '')]
    for i in range(len(y_tr)):
        ev = [(t, 0, f'{A_tr[i, k]:g}', '.') for k, t in enumerate(TT)] + [(s, 1, '.', f'{y_tr[i, j]:.6f}') for j, s in enumerate(SAMP)]
        arm = f',{int(A_tr[i, 0] == 800)}' if cli.arm_col else ''
        lines += [f'{i + 1},{t:g},{a},{y}{arm}' for t, _, a, y in sorted(ev, key=lambda e: (e[0], e[1]))]
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    open(cli.out, 'w').write('\n'.join(lines) + '\n')
    print(f'wrote {cli.out}: {len(y_tr)} patients, {len(SAMP)} samples each, simulator '
          f'{"linear" if KM_TRUE is None else f"saturable Km {KM_TRUE:g}"}', flush=True)
    sys.exit(0)

torch.set_default_dtype(torch.float64)
DT = G.DT
NAMES = ('ka', 'ke', 'V')
if cli.pop == 'true':
    mu = torch.tensor(np.log([G.KA, G.KE, G.V])); Om = torch.tensor(np.outer([G.SD] * 3, [G.SD] * 3) * G.CORR)
    SIG = float(G.ADD_E); KM_HAT = KM_TRUE
    assert cli.struct == 'lin' or KM_TRUE is not None, '--struct mm with --pop true needs a saturable simulator'
    pop_desc = 'TRUE population (simulator)'
else:
    e = json.load(open(cli.pop))['estimates']
    mu = torch.tensor(np.log([e[f'{n}_pop'] for n in NAMES])); om = np.array([e[f'omega_{n}'] for n in NAMES]); C = np.eye(3)
    for i in range(3):
        for j in range(i + 1, 3):
            k = next(k for k in (f'corr_{NAMES[i]}_{NAMES[j]}', f'corr_{NAMES[j]}_{NAMES[i]}') if k in e)
            C[i, j] = C[j, i] = e[k]
    Om = torch.tensor(np.outer(om, om) * C); SIG = float(e['a']); KM_HAT = float(e['Km_pop']) if cli.struct == 'mm' else None
    pop_desc = f'SAEM estimates ({cli.pop})'
    # arm covariate (the "patient route" as a population model): log(theta_i) = log(theta_pop) + beta * ARM800_i + eta_i
    BETA_ARM = torch.tensor([float(e.get(f'beta_{n}_ARM800', 0.0)) for n in NAMES])
    if BETA_ARM.abs().sum() > 0:
        print(f'  arm covariate (800 vs 400 mg) on log(ka, ke, V): {BETA_ARM.numpy().round(4).tolist()}', flush=True)
Om_inv = torch.linalg.inv(Om)
if cli.pop == 'true':
    BETA_ARM = torch.zeros(3)
print(f'structural model {cli.struct}{"" if KM_HAT is None or cli.struct == "lin" else f" (Km {KM_HAT:.1f})"}; population: {pop_desc}\n'
      f'  pop (ka, ke, V) {np.exp(mu.numpy()).round(4).tolist()}  omega {np.sqrt(np.diag(Om.numpy())).round(3).tolist()}  '
      f'sd {SIG:.2f}  | data simulator {"linear" if KM_TRUE is None else f"saturable Km {KM_TRUE:g}"}', flush=True)
obs_idx = np.array([int(np.argmin(np.abs(t_eval - s))) for s in SAMP])


def dose_steps(times):
    return [int(np.argmax(t_eval >= t)) for t in times]


def conc_lin(eta, times, amounts, idx):
    """closed form of the simulator's explicit-Euler linear update (see mapbe_regimen.py)"""
    ka, ke, v = torch.exp(eta).unbind(-1)
    ke = torch.where((ka - ke).abs() < 1e-7, ke + 1e-7, ke)
    a, gg = 1 - ke * DT, 1 - ka * DT
    n = torch.tensor(idx, dtype=torch.float64)
    c = v[:, None] * a[:, None] ** n[None, :]
    for kd, k in enumerate(dose_steps(times)):
        m = (n - k).clamp(min=0)[None, :]
        c = c + (n[None, :] >= k) * torch.tensor(amounts[:, kd])[:, None] * ka[:, None] * (a[:, None] ** m - gg[:, None] ** m) / (ka - ke)[:, None]
    return c


def conc_mm(eta, times, amounts, idx, km):
    """the simulator's explicit Euler loop (dt 0.01, clipping at 0) with saturable elimination, Km = km"""
    ka, ke, v = torch.exp(eta).unbind(-1); Vm = ke * km
    gut = torch.zeros_like(v); cen = v.clone(); A = torch.tensor(amounts); steps = dose_steps(times); out = {}
    want = set(int(i) for i in idx)
    for n in range(int(max(idx)) + 1):
        for kd, k in enumerate(steps):
            if k == n:
                gut = gut + A[:, kd]
        if n in want:
            out[n] = cen
        gut, cen = (gut - ka * gut * DT).clamp(min=0), (cen + (ka * gut - Vm * cen / (km + cen)) * DT).clamp(min=0)
    return torch.stack([out[int(i)] for i in idx], 1)


# exactness of the likelihood code against the DATA simulator (true parameters, training regimen)
times0, A0 = regs['train']
ref = (conc_lin(torch.tensor(np.log(P_te[:20])), times0, A0[:20], np.arange(len(t_eval))) if KM_TRUE is None else
       conc_mm(torch.tensor(np.log(P_te[:20])), times0, A0[:20], np.arange(len(t_eval)), KM_TRUE)).numpy()
_, c_sim = G.simulate(P_te[:20], times0, A0[:20])
rel = np.abs(ref - c_sim).max() / np.abs(c_sim).max()
print(f'likelihood code vs their simulator: max |diff| / max conc = {rel:.2e}', flush=True)
assert rel < 1e-9

CONC = conc_lin if cli.struct == 'lin' else (lambda eta, times, A, idx: conc_mm(eta, times, A, idx, KM_HAT))


def simulate_struct(theta, times, A):
    """prediction with the CHOSEN structural model (G.simulate with the clinician's Km, or linear)"""
    old = G.KM; G.KM = None if cli.struct == 'lin' else KM_HAT
    try:
        return G.simulate(theta, times, A)
    finally:
        G.KM = old


Y = torch.tensor(y_te)
MU_I = mu + torch.tensor((A_te[:, 0] == 800).astype(float))[:, None] * BETA_ARM      # prior mean per test patient (its observed arm)
eta = MU_I.clone().requires_grad_(True)
opt = torch.optim.LBFGS([eta], lr=1, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-12, line_search_fn='strong_wolfe')


def objective():
    opt.zero_grad()
    c = CONC(eta, times0, A_te, obs_idx); d = eta - MU_I
    f = ((Y - c) ** 2).sum() / (2 * SIG ** 2) + 0.5 * torch.einsum('bi,ij,bj->', d, Om_inv, d)
    f.backward()
    return f


for _ in range(5):
    opt.step(objective)
gnorm = float(eta.grad.abs().max()); theta = torch.exp(eta.detach()).numpy()
print(f'MAP done: max |gradient| {gnorm:.2e}; median EBE (ka, ke, V) observed on 400 mg {np.median(theta[0::2], 0).round(3).tolist()}, '
      f'on 800 mg {np.median(theta[1::2], 0).round(3).tolist()} (true medians {np.median(P_te[0::2], 0).round(3).tolist()} / '
      f'{np.median(P_te[1::2], 0).round(3).tolist()})', flush=True)

res = dict(model=f'MAP-BE, structural model {cli.struct}, population {pop_desc}', struct=cli.struct, pop=cli.pop,
           beta_arm800=BETA_ARM.tolist(), pop_values=dict(ka=float(np.exp(mu[0])), ke=float(np.exp(mu[1])), V=float(np.exp(mu[2])),
                           omega=np.sqrt(np.diag(Om.numpy())).tolist(), sd=SIG, Km=KM_HAT if cli.struct == 'mm' else None),
           simulator_km=KM_TRUE, N_per_arm=cli.n, design=cli.design, seed=cli.seed, n_samples_per_patient=len(SAMP),
           max_abs_grad=gnorm, ebe_median_by_arm={'400': np.median(theta[0::2], 0).tolist(), '800': np.median(theta[1::2], 0).tolist()},
           individual={})
for name in G.REGIMENS:
    times, A = regs[name]
    _, c_hat = simulate_struct(theta, times, A)
    pred = np.stack([np.interp(EG, t_eval, c) for c in c_hat]); tru = np.stack([np.interp(EG, t_eval, c) for c in truth[name]])
    ap_, at_ = np.trapezoid(pred, EG, axis=1), np.trapezoid(tru, EG, axis=1); r = ap_ / at_ - 1
    nrmse = np.sqrt(np.mean((pred - tru) ** 2, 1)) / np.mean(tru, 1)
    res['individual'][name] = dict(auc_abs_median=100 * float(np.median(np.abs(r))), nrmse_median=100 * float(np.median(nrmse)),
                                   auc_bias_median=100 * float(np.median(r)), auc_err_per_patient=[float(x) for x in r],
                                   auc_pred_per_patient=[float(x) for x in ap_],
                                   bias_by_observed_arm={'400': 100 * float(np.median(r[0::2])), '800': 100 * float(np.median(r[1::2]))})
    print(f'{name:8s} |AUC err| {res["individual"][name]["auc_abs_median"]:5.2f}%  bias {res["individual"][name]["auc_bias_median"]:+6.2f}%  '
          f'(observed on 400: {100 * np.median(r[0::2]):+6.2f}% | on 800: {100 * np.median(r[1::2]):+6.2f}%)', flush=True)
k = np.log(850 / 350)
ep = np.log(np.array(res['individual']['amt850']['auc_pred_per_patient']) / np.array(res['individual']['amt350']['auc_pred_per_patient'])) / k
res['elasticity'] = dict(pred_median=float(np.median(ep)))
print(f'dose elasticity 350 -> 850 mg: median {np.median(ep):.3f}', flush=True)
os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
json.dump(res, open(cli.out, 'w'), indent=1)
print(f'MAPBE_DONE {(time.time() - t0) / 60:.1f} min', flush=True)
