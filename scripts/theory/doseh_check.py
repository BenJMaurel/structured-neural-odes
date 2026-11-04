#!/usr/bin/env python3
"""Variant B (--dose-history) structural checks on trained checkpoints.

    doseh_check.py <cohort> <scale-from cohort> <ckpt> [<ckpt> ...] [--n 300]

Posterior means of the first --n test patients of <cohort>, normalisation pinned to <scale-from>. Per checkpoint:
  1. conservation   max |P_phys (z_t - z_phys)| over the 7-dose history and the window (exact by construction: ~1e-5)
  2. drug-free      dose 0: max |z_t - z_phys| after 7 intervals; max Re eigenvalue of the drug block of the flow
                    Jacobian at the drug-free point (> 0 = unstable equilibrium; NOT enforced, only initialised stable)
  3. steady state   window AUC after 7 vs 14 / 28 doses at the patient's visit-1 dose, and the per-interval contraction
                    factor ||s_{k+1} - s_k|| / ||s_k - s_{k-1}|| of the post-dose state (< 1 = contracting). The cohort's
                    own design is the 7th dose from an empty body, NOT steady state (gen_tacro_confound: MM patients
                    are often far from it), so 7 != 28 is not an error per se -- it says what the model does past the data.
  4. dose response  window AUC on a 0.25-12 mg grid: share of patients with AUC non-decreasing in dose, median
                    AUC(2d)/AUC(d) (2 = linear; MM elimination gives > 2), AUC at 0.25 / 0.5 mg (the 'lo' cohort range).
"""
import os, sys, argparse
import numpy as np
import torch
sys.path.insert(0, os.getcwd())
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film, set_static_hematocrit
from probe_entanglement import load_film_model
from test_dose_cond import unscale_data, calculate_auc

ap = argparse.ArgumentParser()
ap.add_argument('cohort'); ap.add_argument('scale_from'); ap.add_argument('ckpts', nargs='+')
ap.add_argument('--n', type=int, default=300)
cli = ap.parse_args()
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
dev = torch.device('cpu')
GRID = np.array([0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12])


def window_auc(model, z0, doses_mg, n_doses, tp, static, scaler, dose_max):
    dh = model.dose_hist
    old = dh.n_doses
    dh.n_doses = n_doses
    try:
        sol = model._dose_history_traj(z0, torch.as_tensor(doses_mg / dose_max, dtype=z0.dtype), tp, form=static[:, 1])
    finally:
        dh.n_doses = old
    conc = unscale_data(model.decoder(sol).mean(0).squeeze(-1), scaler)
    return calculate_auc(conc, tp.numpy(), static)


for ck in cli.ckpts:
    print(f'=== {ck}')
    if not os.path.exists(ck):
        print('   missing'); continue
    a = torch.load(ck, map_location='cpu', weights_only=False)['args']
    set_static_hematocrit(int(getattr(a, 'static_dim', 3)) >= 4)
    model, args = load_film_model(ck, dev)
    if getattr(model, 'dose_hist', None) is None:
        print('   not a --dose-history checkpoint'); continue
    pin = extract_gen_tac_film(file_path=[f'results/exp_film_run/{cli.scale_from}/virtual_cohort_film_train.csv'])[1]
    tr = f'results/exp_film_run/{cli.cohort}/virtual_cohort_film_train.csv'
    te = f'results/exp_film_run/{cli.cohort}/virtual_cohort_film_test.csv'
    data, scaler = extract_gen_tac_film(file_path=[tr, te], scale=pin)
    train_ids = set(extract_gen_tac_film(file_path=[tr], scale=pin)[0].keys())
    ids = [k for k in data if k not in train_ids][:cli.n]
    batch = collate_fn_tacro_film([TacroFilmDataset({k: data[k] for k in ids})[i] for i in range(len(ids))], dev)
    dose_max = float(pin[2]) if len(pin) > 2 else 8.0
    os_, dh, ode = model.orth_split, model.dose_hist, model.diffeq_solver.ode_func
    tp = torch.linspace(0, 24, 100)
    with torch.no_grad():
        enc_in, st = model._encoder_input(batch['observed_data_v1'], batch['dose_v1'], batch['static_v1'])
        mu, _ = model.encoder_z0(enc_in, batch['observed_tp_v1'], static=st, run_backwards=True)
        z0 = mu[:1]
        zp = z0 @ os_.P_phys()
        d1 = batch['dose_v1'].numpy() * dose_max
        print(f'   {len(ids)} test patients, dose_max {dose_max:g} mg, kappa per drug dim {np.round(dh.kappa().numpy(), 3).tolist()}, '
              f'|kick| at 1 mg {float(dh.kick().norm()) / dose_max:.3f}')
        # 1. conservation
        form = batch['static_v1'][:, 1]
        pg = getattr(dh, 'interval_prograf', 0.0)
        print(f'   dosing interval: {"12/24 h by formulation" if pg > 0 else f"{dh.interval:g} h for everyone"}; '
              f'Prograf share {100 * float(form.mean()):.0f}%')
        sol = model._dose_history_traj(z0, batch['dose_v1'], tp, form=form)
        print(f'1. conservation: max |P_phys(z_t - z_phys)| = {float(((sol - zp.unsqueeze(2)) @ os_.P_phys()).abs().max()):.2e}')
        # 2. drug-free equilibrium
        sol0 = model._dose_history_traj(z0, torch.zeros_like(batch['dose_v1']), tp, form=form)
        drift = (sol0 - zp.unsqueeze(2)).norm(dim=-1).amax(dim=-1).squeeze(0).numpy()
    Qd = os_.Q_drug().detach()
    eigs = []
    for i in range(len(ids)):
        y = zp[0, i].detach().clone()
        ode._anchor = None
        J = torch.autograd.functional.jacobian(lambda u: ode(0., u.reshape(1, 1, -1)).reshape(-1), y)
        eigs.append(float(torch.linalg.eigvals(Qd.t() @ J @ Qd).real.max()))
    eigs = np.array(eigs)
    print(f'2. drug-free: dose 0 drift over history+window median {np.median(drift):.2e} max {drift.max():.2e}; '
          f'max Re eig at u=0 median {np.median(eigs):+.3f} [{eigs.min():+.3f}, {eigs.max():+.3f}], unstable {100 * np.mean(eigs > 0):.1f}%')
    with torch.no_grad():
        # 3. steady state
        a7, a14, a28 = (window_auc(model, z0, d1, n, tp, batch['static_v1'], scaler, dose_max) for n in (7, 14, 28))
        r14, r28 = np.abs(a14 / a7 - 1), np.abs(a28 / a7 - 1)
        print(f'3. steady state at the visit-1 dose: |AUC_14/AUC_7 - 1| median {100 * np.median(r14):.1f}% (>10%: {100 * np.mean(r14 > .1):.0f}%), '
              f'|AUC_28/AUC_7 - 1| median {100 * np.median(r28):.1f}% (>10%: {100 * np.mean(r28 > .1):.0f}%)')
        kick = torch.as_tensor(d1 / dose_max, dtype=z0.dtype).reshape(1, -1, 1) * dh.kick()
        z, post, ode._anchor = zp.clone(), [], ode.gradient_net(zp)
        if pg > 0:   # per-patient interval, as in LatentODE._dose_history_traj
            ode._tscale = torch.where(form.reshape(1, -1, 1) > 0.5, pg, dh.interval).to(z0)
            iv = torch.tensor([0.0, 1.0])
        else:
            iv = torch.tensor([0.0, dh.interval])
        for k in range(28):
            z = z + kick; post.append(z.clone())
            z = model.diffeq_solver(z, iv)[:, :, -1, :]
        ode._anchor = None; ode._tscale = None
        steps = torch.stack([(post[k + 1] - post[k]).norm(dim=-1).squeeze(0) for k in range(27)])
        # ratios of steps below 1e-6 are solver noise (dopri5 rtol 1e-3, atol 1e-4), not dynamics
        fac = torch.where(steps[:-1] > 1e-6, steps[1:] / steps[:-1].clamp(min=1e-12), torch.full_like(steps[1:], float('nan'))).numpy()
        md = lambda x: (f'{np.nanmedian(x):.3f}' if np.isfinite(x).any() else 'converged')
        print(f'   contraction factor per interval, median over patients at dose 2-3 / 7-8 / 20-27 (steps > 1e-6 only): '
              f'{md(fac[0:2])} / {md(fac[5:7])} / {md(fac[18:])}; '
              f'patients still moving > 1e-3 at dose 28: {100 * np.mean(steps[-1].numpy() > 1e-3):.0f}%')
        # 4. dose response
        A = np.stack([window_auc(model, z0, np.full(len(ids), g), 7, tp, batch['static_v1'], scaler, dose_max) for g in GRID], 1)
        mono = np.all(np.diff(A, axis=1) >= -1e-6, axis=1)
        pairs = [(i, list(GRID).index(2 * g)) for i, g in enumerate(GRID) if 2 * g in GRID]
        ratio = np.median(np.stack([A[:, j] / A[:, i] for i, j in pairs], 1), axis=0)
        print(f'4. dose response (7 doses): AUC non-decreasing in dose for {100 * mono.mean():.1f}% of patients; median AUC(2d)/AUC(d) '
              + ', '.join(f'{GRID[i]:g}->{GRID[j]:g} mg {r:.2f}' for (i, j), r in zip(pairs, ratio))
              + f'; median AUC at 0.25 / 0.5 / 1 mg {np.median(A[:, 0]):.2f} / {np.median(A[:, 1]):.2f} / {np.median(A[:, 2]):.2f} ng.h/mL')
