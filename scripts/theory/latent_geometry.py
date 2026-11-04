#!/usr/bin/env python3
"""Latent geometry of a trained OT-FiLM model: is physiology in the orthogonal complement of the FiLM (dose) axis, and does
it stay there while the latent flow runs?, design of a unified dose-action model.)

For every held-out patient: posterior means z0(v1), z0(v2) from the two visits; FiLM displacements z0' - z0 for the
patient's d1 to every in-range dose 1..8 mg; the latent trajectory z_t from z0(v1) over the dosing interval (100 points,
Prograf cut at 12 h as in test_film_matched).

Measures (all in the model's own latent coordinates, prior N(0, I)):
  FiLM axis      uncentred SVD of the grid displacements: energy in the top 1/2/3 directions; axis v = top direction.
                 Share of the displacement from beta vs from (gamma - 1) * z0.
  dimension      uncentred SVD of the within-interval motion z_t - z0; and of the joint set [FiLM displacements, motion]
                 (each block scaled to unit total energy) = the empirical 'drug subspace'.
  identity ratio for a projection P:  E||P(a - b)||^2 / (2 tr Cov(P z0)) -- 0 = same patient, 1 = as different as two
                 random patients. Applied to (z_t, z0) at each t = DRIFT, and to (z0(v2), z0(v1)) = visit consistency.
                 Posterior noise floor: tr(P Sigma_post P) / tr Cov(P z0).
  probes         ridge (5-fold CV) of true physiology and of d1 from P z0; the same fitted probe applied to P z_t
                 (physiology read at time t with the map learnt at t = 0).
Projections: full z, complement of the FiLM axis (Idea 2 as stated), complement of the top-k joint drug subspace.

    latent_geometry.py <cohort> <scale_from> <out_prefix> <ckpt> [<ckpt> ...]
"""
import sys, os, json, numpy as np, pandas as pd, torch
sys.path.insert(0, '.')
from torch.utils.data import DataLoader
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film, set_static_hematocrit
from probe_entanglement import load_film_model, encode_mu

R = './results/exp_film_run'
GRID_MG = np.arange(1.0, 9.0)
PROBE_T = (1.0, 3.0, 6.0, 12.0)
dev = torch.device('cpu')


def collect(model, cohort, scale_from):
    pin = extract_gen_tac_film(file_path=[f'{R}/{scale_from}/virtual_cohort_film_train.csv'])[1]
    dmax = float(pin[2]) if len(pin) > 2 else 8.0
    tr_csv, te_csv = f'{R}/{cohort}/virtual_cohort_film_train.csv', f'{R}/{cohort}/virtual_cohort_film_test.csv'
    data, _ = extract_gen_tac_film(file_path=[tr_csv, te_csv], scale=pin)
    tr_ids = set(extract_gen_tac_film(file_path=[tr_csv], scale=pin)[0].keys())
    data = {k: v for k, v in data.items() if k not in tr_ids}
    loader = DataLoader(TacroFilmDataset(data), batch_size=256, shuffle=False,
                        collate_fn=lambda x: collate_fn_tacro_film(x, dev))
    tp = torch.linspace(0.0, 24.0, 100)
    out = {k: [] for k in ('ids', 'z1', 'z2', 's1', 'disp', 'disp_beta', 'disp_gamma', 'traj', 'prograf', 'd1', 'd2')}
    grid = torch.tensor(GRID_MG / dmax, dtype=torch.float32)
    with torch.no_grad():
        for b in loader:
            tw, st = model._encoder_input(b['observed_data_v1'], b['dose_v1'], b['static_v1'])
            mu1, sd1 = model.encoder_z0(tw, b['observed_tp_v1'], static=st, run_backwards=True)
            z1 = mu1.squeeze(0)
            z2 = encode_mu(model, b['observed_data_v2'], b['observed_tp_v2'], b['dose_v2'], b.get('static_v2', b['static_v1']))
            B = z1.shape[0]
            D, Db, Dg = [], [], []
            for g in grid:
                d2 = g.repeat(B)
                c = model._film_context(torch.stack([b['dose_v1'], d2], 1).unsqueeze(0), z1.unsqueeze(0))
                gam, bet = model._film_maps(c, b['dose_v1'], d2, 1)
                gam, bet = gam.squeeze(0), bet.squeeze(0)
                D.append(model._film_apply(z1, gam, bet) - z1); Db.append(bet); Dg.append(z1 * (gam - 1))
            traj = model.diffeq_solver(z1.unsqueeze(0), tp).squeeze(0)       # [B, T, L]
            out['ids'] += list(b['patient_ids']); out['z1'].append(z1); out['z2'].append(z2); out['s1'].append(sd1.squeeze(0).abs())
            out['disp'].append(torch.stack(D, 1)); out['disp_beta'].append(torch.stack(Db, 1)); out['disp_gamma'].append(torch.stack(Dg, 1))
            out['traj'].append(traj); out['prograf'].append(b['static_v1'][:, 1] > 0.5)
            out['d1'].append(b['dose_v1'] * dmax); out['d2'].append(b['dose_v2'] * dmax)
    res = {k: (v if k == 'ids' else torch.cat(v).numpy()) for k, v in out.items()}
    res['tp'] = tp.numpy(); res['dmax'] = dmax
    return res


def targets(cohort, ids, d1):
    T = pd.read_csv(f'{R}/{cohort}/confound_truth.csv').set_index('ID').loc[[int(i) for i in ids]]
    y = {}
    for p in ('Vmax', 'Km', 'Vc', 'Vp', 'Q', 'Ktr', 'CL'):
        if p in T.columns:
            y[f'log {p}' + (' (inert, control)' if p == 'CL' else '')] = np.log(T[p].values.astype(float))
    if 'log Vc' not in y and 'confound_par' in T.columns and (T.confound_param == 'Vc').all():
        y['log Vc'] = np.log(T.confound_par.values.astype(float))
    if 'log CL (inert, control)' not in y:
        y['log CL (inert, control)'] = np.log(T.CL_base.values.astype(float))
    y['CYP expresser (covariate input)'] = (T.CYP == 'expresser').values.astype(float)
    y['d1 (dose, must be absent)'] = np.asarray(d1, float)
    return y


def ratio(P, A, Bm, ref):
    """E||P(A - B)||^2 / (2 tr Cov(P ref)); rows are patients."""
    den = 2.0 * np.trace(np.cov((ref @ P.T).T))
    return float(np.mean(np.sum(((A - Bm) @ P.T) ** 2, 1)) / den)


def _ridge():
    return Pipeline([('s', StandardScaler()), ('m', RidgeCV(alphas=np.logspace(-3, 3, 13)))])


def cv_r2(X, t, seed=0):
    pred = np.zeros_like(t)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        pred[te] = _ridge().fit(X[tr], t[tr]).predict(X[te])
    return float(r2_score(t, pred))


def probes(P, z0, zt_by_t, y, seed=0):
    """'tK' = probe fitted on P z0 applied to P z_t (fixed physiology readout); 'rK' = probe refitted at time t."""
    out = {}
    X0 = z0 @ P.T
    for name, t in y.items():
        pred0 = np.zeros_like(t); pred_t = {k: np.zeros_like(t) for k in zt_by_t}
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(X0):
            m = _ridge().fit(X0[tr], t[tr])
            pred0[te] = m.predict(X0[te])
            for k, Z in zt_by_t.items():
                pred_t[k][te] = m.predict(Z[te] @ P.T)
        out[name] = {'t0': float(r2_score(t, pred0)), **{f't{k:g}': float(r2_score(t, pred_t[k])) for k in zt_by_t},
                     **{f'r{k:g}': cv_r2(Z @ P.T, t, seed) for k, Z in zt_by_t.items()}}
    return out


def analyse(r, y):
    z1, z2, traj, tp, prog = r['z1'], r['z2'], r['traj'], r['tp'], r['prograf']
    L = z1.shape[1]; I = np.eye(L)
    valid = np.where(prog[:, None], tp[None] <= 12.0 + 1e-6, True)            # [N, T]
    disp = r['disp'].reshape(-1, L)
    keep = np.abs(r['d1'][:, None] - GRID_MG[None]).reshape(-1) > 1e-6       # drop the identity pair d2 = d1
    disp = disp[keep]
    motion = (traj - z1[:, None])[valid]                                       # [rows, L]
    sv = lambda X: np.linalg.svd(X, full_matrices=False)
    _, sd, Vd = sv(disp); ed = np.cumsum(sd ** 2) / np.sum(sd ** 2)
    _, sm, Vm = sv(motion); em = np.cumsum(sm ** 2) / np.sum(sm ** 2)
    J = np.vstack([disp / np.sqrt(np.sum(disp ** 2)), motion / np.sqrt(np.sum(motion ** 2))])
    _, sj, Vj = sv(J); ej = np.cumsum(sj ** 2) / np.sum(sj ** 2)
    v = Vd[0]
    enc = z2 - z1
    eb = float(np.mean(np.sum(r['disp_beta'].reshape(-1, L)[keep] ** 2, 1)))
    eg = float(np.mean(np.sum(r['disp_gamma'].reshape(-1, L)[keep] ** 2, 1)))
    film_act = None
    nrm = lambda X: float(np.median(np.linalg.norm(X, axis=1)))
    out = {'n_patients': int(len(z1)), 'latent_dim': L,
           'norms': {'z0_minus_mean': nrm(z1 - z1.mean(0)), 'sqrt_trcov_z0': float(np.sqrt(np.trace(np.cov(z1.T)))),
                     **{f'move_{t:g}h': nrm((traj[:, int(np.argmin(np.abs(tp - t)))] - z1)[valid[:, int(np.argmin(np.abs(tp - t)))]]) for t in (1, 3, 6, 12, 24)},
                     'enc_disp': nrm(enc), 'film_disp': nrm(disp)},
           'film_disp_energy_top123': [float(x) for x in ed[:3]],
           'film_disp_beta_vs_gamma_energy': [eb, eg],
           'enc_disp_share_on_axis': float(np.sum((enc @ v) ** 2) / np.sum(enc ** 2)),
           'enc_disp_share_top3_film': float(np.sum((enc @ Vd[:3].T) ** 2) / np.sum(enc ** 2)),
           'motion_energy_cum': [float(x) for x in em],
           'joint_energy_cum': [float(x) for x in ej],
           'axis_share_in_motion_top2': float(np.sum((Vm[:2] @ v) ** 2)),
           'axis_share_in_motion_top3': float(np.sum((Vm[:3] @ v) ** 2)),
           'motion_share_on_axis': float(np.sum((motion @ v) ** 2) / np.sum(motion ** 2))}
    projs = {'full z': I, 'complement of FiLM axis (Idea 2)': I - np.outer(v, v)}
    for k in (1, 2, 3, 4, 6):
        projs[f'complement of joint drug subspace k={k}'] = I - Vj[:k].T @ Vj[:k]
    it = {t: int(np.argmin(np.abs(tp - t))) for t in PROBE_T}
    zt = {t: traj[:, it[t]] for t in PROBE_T}
    zt = {t: Z for t, Z in zt.items()}
    var_post = np.mean(r['s1'] ** 2, 0)
    out['proj'] = {}
    for name, P in projs.items():
        den = 2.0 * np.trace(np.cov((z1 @ P.T).T))
        dr = np.array([np.sum((((traj[:, j] - z1) @ P.T) ** 2)[valid[:, j]], 1).mean() / den for j in range(len(tp))])
        # probes at t <= 12 h are valid for both formulations
        com = {}
        for t in PROBE_T + (24.0,):
            j = it.get(t, len(tp) - 1); m = valid[:, j]
            D = ((traj[:, j] - z1) @ P.T)[m]; mu = D.mean(0)
            com[f'{t:g}'] = {'common_share': float(np.sum(mu ** 2) / np.mean(np.sum(D ** 2, 1))),
                             'specific_ratio': float(np.mean(np.sum((D - mu) ** 2, 1)) / den),
                             'spread_growth': float(np.trace(np.cov((traj[m, j] @ P.T).T)) / np.trace(np.cov((z1[m] @ P.T).T)))}
        out['proj'][name] = {
            'decomp': com,
            'd2_from_v2': {'r2': cv_r2(z2 @ P.T, r['d2']), 'r2_d1': cv_r2(z2 @ P.T, r['d1'])},
            'drift_max': float(dr.max()), 'drift_at': {f'{t:g}': float(dr[it[t]]) for t in PROBE_T},
            'drift_24h_advagraf': float(dr[-1]),
            'visit_ratio': ratio(P, z2, z1, z1),
            'noise_floor': float(np.trace(P @ np.diag(var_post) @ P.T) / np.trace(np.cov((z1 @ P.T).T))),
            'probes': probes(P, z1, zt, y)}
    return out, {'v': v.tolist(), 'Vj': Vj[:4].tolist()}


def main():
    cohort, scale_from, prefix = sys.argv[1:4]
    ckpts = sys.argv[4:]
    res = []
    for ck in ckpts:
        a = torch.load(ck, map_location='cpu', weights_only=False)['args']
        set_static_hematocrit(int(getattr(a, 'static_dim', 3)) >= 4)
        model, _ = load_film_model(ck, dev)
        r = collect(model, cohort, scale_from)
        y = targets(cohort, r['ids'], r['d1'])
        o, axes = analyse(r, y); o['ckpt'] = ck
        res.append(o); print(f'done {ck}: {o["n_patients"]} patients', flush=True)
    json.dump(res, open(f'{prefix}.json', 'w'), indent=1)
    txt = report(res, cohort)
    print(txt); open(f'{prefix}.txt', 'w').write(txt + '\n'); print('GEOMETRY_DONE', prefix)


def report(res, cohort):
    f = lambda xs: f'{np.mean(xs):6.3f} [{min(xs):.3f}-{max(xs):.3f}]' if len(xs) > 1 else f'{xs[0]:6.3f}'
    L = [f'=== latent geometry, cohort {cohort}, {len(res)} checkpoint(s), {res[0]["n_patients"]} test patients, latent dim {res[0]["latent_dim"]}',
         '    numbers: mean [min-max] over seeds', '',
         '-- FiLM (dose) displacement, grid d1 -> 1..8 mg',
         f'   energy in top 1 / 2 / 3 directions: ' + ' / '.join(f([o['film_disp_energy_top123'][i] for o in res]) for i in range(3)),
         f'   mean ||beta||^2 vs ||(gamma-1)*z0||^2: ' + f([o['film_disp_beta_vs_gamma_energy'][0] for o in res]) + ' vs ' + f([o['film_disp_beta_vs_gamma_energy'][1] for o in res]),
         f'   encoder displacement z0(v2)-z0(v1): share on the FiLM axis {f([o["enc_disp_share_on_axis"] for o in res])}, on the top-3 FiLM directions {f([o["enc_disp_share_top3_film"] for o in res])}',
         '', '-- within-interval latent motion z_t - z0',
         '   cumulative energy, k = 1..5: ' + ' / '.join(f([o['motion_energy_cum'][i] for o in res]) for i in range(5)),
         f'   share of the motion along the FiLM axis: {f([o["motion_share_on_axis"] for o in res])}',
         f'   FiLM axis inside the top-2 / top-3 motion directions (squared cosine): {f([o["axis_share_in_motion_top2"] for o in res])} / {f([o["axis_share_in_motion_top3"] for o in res])}',
         '   joint drug subspace [FiLM displacement + motion], cumulative energy k = 1..5: ' + ' / '.join(f([o['joint_energy_cum'][i] for o in res]) for i in range(5)),
         '', '-- raw sizes (median Euclidean norms): ' + ', '.join(f'{k} {np.mean([o["norms"][k] for o in res]):.3g}' for k in res[0]['norms']),
         '', '-- identity ratio (0 = same patient, 1 = as different as two random patients)']
    for name in res[0]['proj']:
        P = [o['proj'][name] for o in res]
        L += [f'   [{name}]',
              f'      drift z_t vs z0: max {f([p["drift_max"] for p in P])}; at 1/3/6/12 h ' + ' / '.join(f'{np.mean([p["drift_at"][k] for p in P]):.3f}' for k in ('1', '3', '6', '12')) + f'; Advagraf 24 h {np.mean([p["drift_24h_advagraf"] for p in P]):.3f}',
              f'      visit 1 vs visit 2 (same patient, other dose): {f([p["visit_ratio"] for p in P])};  posterior noise floor {f([p["noise_floor"] for p in P])}',
              '      drift decomposition at 1/3/6/12/24 h: common-mode share ' + ' / '.join(f'{np.mean([p["decomp"][k]["common_share"] for p in P]):.2f}' for k in ('1', '3', '6', '12', '24'))
              + ';  patient-specific ratio ' + ' / '.join(f'{np.mean([p["decomp"][k]["specific_ratio"] for p in P]):.2f}' for k in ('1', '3', '6', '12', '24'))
              + ';  spread growth ' + ' / '.join(f'{np.mean([p["decomp"][k]["spread_growth"] for p in P]):.2f}' for k in ('1', '3', '6', '12', '24')),
              f'      dose-free test on z0(v2): R^2 of d2 {f([p["d2_from_v2"]["r2"] for p in P])}  (d1 via physiology {f([p["d2_from_v2"]["r2_d1"] for p in P])})',
              '      probe R^2 at t = 0 / 1 / 3 / 6 / 12 h -- fixed (fit at t=0) || refit at t:']
        for tgt in P[0]['probes']:
            L.append(f'        {tgt:34s} ' + ' / '.join(f'{np.mean([p["probes"][tgt][k] for p in P]):6.2f}' for k in ('t0', 't1', 't3', 't6', 't12'))
                     + '  ||  ' + ' / '.join(f'{np.mean([p["probes"][tgt][k] for p in P]):5.2f}' for k in ('r1', 'r3', 'r6', 'r12')))
    return '\n'.join(L)


if __name__ == '__main__':
    main()
