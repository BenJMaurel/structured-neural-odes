#!/usr/bin/env python3
"""MAP-BE on the scenario-4 test patients under a LINEAR-elimination population model estimated by Monolix
(scripts/monolix/fit_s4_select.R ... lin), then the Visit-2 counterfactual: the 'NLME with the wrong structure' row
next to the true-structure references of ebe_popmodel.py (2026-10-02).

Same protocol as ebe_popmodel.py: one fit to the 3 Visit-1 points (0, 1, 3 h; the encoder's inputs), then predictions
for the same patient's Visit-1 interval and Visit-2 dose in range / below / above (cohorts COHORT, _lo, _hi); same
steady-state dosing (oracle_floor.BatchPK, NBR_SS doses), same Levenberg-Marquardt, same output columns, so
ndm_tables.ebe_cell() scores it. Only the structure differs: BatchPK(scenario=1) = 3 transits + 2 compartments +
first-order elimination k = CL/Vc. Modeller covariates as fitted (variant l1): Ktr ~ ST, Vc ~ ST,
CL ~ log(HT/35) + CYP; diagonal omega; combined1 error a + b*f. Covariates read from confound_truth.csv.

    EBE_COHORT=<cohort> ebe_popmodel_lin.py <estimates.json> <n> <seed> <out.csv>
"""
import sys, os, json, numpy as np, pandas as pd, torch
sys.path.insert(0, 'scripts/ident'); sys.path.insert(0, 'scripts/monolix'); sys.path.insert(0, '.')
from oracle_floor import BatchPK, NBR_SS, ENC_T
from ebe_s4_batch import dosing
from ebe_popmodel import load, curve, REL
COHORT = os.environ.get('EBE_COHORT', 'confound_vc00_s4_pnoise_win')
FIT = ['Ktr', 'Q', 'Vc', 'Vp', 'CL']


def simulate(theta, dose, form, rel):
    """theta [M,5] natural-scale (FIT order), dose [M] mg -> conc [len(rel), M] ng/mL at steady state (as ebe_s4_batch)."""
    M = theta.shape[0]
    pk = BatchPK(M, formulation=form, hematocrit=35.0, cyp_status='non_expresser', scenario=1)
    pk.individual_params = {n: torch.as_tensor(theta[:, i], dtype=torch.float32) for i, n in enumerate(FIT)}
    pk.dose_mg = torch.as_tensor(dose, dtype=torch.float32)
    dos = dosing(form)
    tgt = torch.as_tensor(np.asarray(rel) + 24.0 * NBR_SS, dtype=torch.float32)
    tp = torch.unique(torch.cat([torch.arange(1.0, 24.0 * NBR_SS, 1.0), torch.tensor(dos[1:]), tgt]))
    tp = tp[tp > 0]
    with torch.no_grad():
        c = pk.simulate(dos, tp, resample=False) * 1000.0
    idx = [int(torch.argmin(torch.abs(tp - t))) + 1 for t in tgt.tolist()]
    return c[idx, :].double().numpy()


def spec(path):
    e = json.load(open(path))['estimates']
    return dict(Ktr=e['Ktr_pop'], bKtr=e['beta_Ktr_ST_1'], Q=e['Q_pop'], Vc=e['Vc_pop'], bVc=e['beta_Vc_ST_1'],
                Vp=e['Vp_pop'], CL=e['CL_pop'], bCLHT=e.get('beta_CL_tHT', 0.0),   # absent in the no-HT fit (22/10)
                 bCLCYP=e['beta_CL_CYP_1'],
                om=np.array([e['omega_Ktr'], e['omega_Q'], e['omega_Vc'], e['omega_Vp'], e['omega_CL']]), a=e['a'], b=e['b'])


def typical(S, st, cyp, ht):
    return np.array([S['Ktr'] * np.exp(S['bKtr'] * st), S['Q'], S['Vc'] * np.exp(S['bVc'] * st), S['Vp'],
                     S['CL'] * np.exp(S['bCLHT'] * np.log(ht / 35.0) + S['bCLCYP'] * cyp)])


def fit_group(y, tv, dose, form, PR, a, b, iters=40, h=1e-3):
    K, P = y.shape[0], len(FIT); eta = np.zeros((K, P)); lam = np.full(K, 1e-2)
    def objective(e):
        f = simulate(tv * np.exp(e), dose, form, ENC_T).T; sd = a + b * f
        return (((y - f) / sd) ** 2).sum(1) + np.einsum('ki,ij,kj->k', e, PR, e)
    obj = objective(eta)
    for it in range(iters):
        pert = eta[:, None, :] + h * np.concatenate([np.zeros((1, P)), np.eye(P)])[None]
        F = simulate((tv[:, None, :] * np.exp(pert)).reshape(-1, P), np.repeat(dose, P + 1), form, ENC_T).T.reshape(K, P + 1, 3)
        f0 = F[:, 0]; J = (F[:, 1:] - f0[:, None]) / h; W = 1.0 / (a + b * f0)
        Jw = J * W[:, None, :]; rw = (y - f0) * W
        A = np.einsum('kip,kjp->kij', Jw, Jw) + PR[None]; g = np.einsum('kip,kp->ki', Jw, rw) - eta @ PR
        step = np.linalg.solve(A + lam[:, None, None] * np.eye(P)[None], g[..., None])[..., 0]
        new = eta + step; nobj = objective(new); ok = nobj < obj
        eta[ok] = new[ok]; obj[ok] = nobj[ok]; lam = np.where(ok, lam * 0.3, lam * 10.0)
        if np.max(np.abs(step[ok]), initial=0) < 1e-5 and it > 5: break
    return eta


def main():
    path, n, seed, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    S = spec(path); PR = np.diag(1.0 / S['om'] ** 2)
    print(f'linear population model {path}: ' + ', '.join(f'{k} {v:.4g}' for k, v in S.items() if k != 'om') +
          f', omega {np.round(S["om"], 3).tolist()}', flush=True)
    C = {r: load(c) for r, c in (('in', COHORT), ('lo', COHORT + '_lo'), ('hi', COHORT + '_hi'))
         if os.path.exists(f'results/exp_film_run/{c}/confound_truth.csv')}
    print(f'cohort {COHORT}: regimes {list(C)}', flush=True)
    T0, O0 = C['in']
    te = T0[T0.split == 'test'].index.values.copy(); np.random.RandomState(seed).shuffle(te); ids = te[:n]
    rows = []
    for form in ('Advagraf', 'Prograf'):
        g = [i for i in ids if T0.loc[i].DRUG == form]
        st = 1.0 if form == 'Prograf' else 0.0
        TV = np.stack([typical(S, st, int(T0.loc[i].CYP == 'expresser'), float(T0.loc[i].HT)) for i in g])
        d1 = np.array([T0.loc[i].d1 for i in g])
        y3 = np.array([[curve(O0, i, 1)[0][int(np.argmin(np.abs(REL - t)))] for t in ENC_T] for i in g])
        eta = fit_group(y3, TV, d1, form, PR, S['a'], S['b']); th = TV * np.exp(eta)
        hi = 12.0 if form == 'Prograf' else 24.0; ta = np.round(np.arange(0.0, hi + 1e-9, 0.02), 4)
        preds = {'v1': (d1, [curve(O0, i, 1) for i in g])}
        for r, (T, O) in C.items():
            preds[r] = (np.array([T.loc[i].d2 for i in g]), [curve(O, i, 2) for i in g])
        base = len(rows); rows += [dict(ID=i, form=form) for i in g]
        for tag, (dd, obs) in preds.items():
            auc = np.trapezoid(simulate(th, dd, form, ta), ta, axis=0); c12 = simulate(th, dd, form, REL).T
            for k in range(len(g)):
                y12, a_true = obs[k]; row = rows[base + k]
                row[f'dose_{tag}'] = dd[k]; row[f'auc_true_{tag}'] = a_true; row[f'auc_{tag}'] = float(auc[k])
                row[f'nrmse_{tag}'] = float(np.sqrt(np.mean((c12[k] - y12) ** 2)) / np.mean(y12))
        for k in range(len(g)):
            for j, nm in enumerate(FIT):
                rows[base + k][f'log_{nm}_ebe'] = float(np.log(TV[k, j]) + eta[k, j])
        print(f'  {form}: {len(g)} patients', flush=True)
    pd.DataFrame(rows).to_csv(out, index=False); print('POPEBE_DONE', out)


if __name__ == '__main__':
    main()
