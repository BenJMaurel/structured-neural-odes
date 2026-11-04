#!/usr/bin/env python3
"""Data for the revised step-3 figure of ndm.tex (). OT-FiLM (step 2), seed-1 checkpoint at ep6000, two-visit in-range test cohort, Advagraf patients
(one 24-h interval), posterior means from visit 1, no training. With the operations of LatentODE.get_reconstruction_extrapolation:
  current dose   z0 -> flow -> concentrations
  transport at 0 z0' = T_{d1->d2}(z0) -> flow                       (what OT-FiLM is trained and evaluated for)
  transport at t the same map T (same gamma, beta) applied to the state reached at t under the current dose, then the flow from t
If the transport acted on a drug state, as a change of the amounts does in a linear compartmental model, applying it at t would give
the same prediction from t on as applying it at the dose. Measured: AUC over [t, 24 h] of 'transport at t' relative to 'transport at 0'
(median over patients, and 10/90 % quantiles), t = 0 ... 18 h; CHECK: at t = 0 the two coincide exactly. One illustrative patient
(median-like: dose ratio d2/d1 closest to 2) for panel a. Writes results/ndm/fig_transport_at_t.npz (+ .txt)."""
import os, sys, glob
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film, set_static_hematocrit
from probe_entanglement import load_film_model
from test_dose_cond import unscale_data

torch.set_num_threads(1); dev = torch.device('cpu')
B = 'confound_vc00_s4_pnoise_win'; T = B + '_n100'; R = './results/exp_film_run'
FROOT = os.environ.get('FIG_FILM_ROOT', 'results/s4_pnoise_win')   # 22/10 opt-in: results/s4_pnoise_win_gpos = Table 2's gamma > 0 OT-FiLM
SEED = os.environ.get('FIG_SEED', '1')                             # 22/10 opt-in: other seeds (Fig. 4b shows every seed)
OUTP = os.environ.get('FIG_OUT', 'results/ndm/fig_transport_at_t')  # 22/10 opt-in: output stem for the per-seed runs
ck = sorted(glob.glob(f'{FROOT}/film_n100_s{SEED}/exp_film_run/{T}/traj/*_ep006000.ckpt'))[0]
a = torch.load(ck, map_location='cpu', weights_only=False)['args']
set_static_hematocrit(int(getattr(a, 'static_dim', 3)) >= 4)
model, _ = load_film_model(ck, dev); model.eval()
assert not getattr(model, 'film_time', False) and getattr(model, 'dose_hist', None) is None
pin = extract_gen_tac_film(file_path=[f'{R}/{T}/virtual_cohort_film_train.csv'])[1]      # --scale-from the training cohort
tr, te = (f'{R}/{B}/virtual_cohort_film_{x}.csv' for x in ('train', 'test'))
data, _ = extract_gen_tac_film(file_path=[tr, te], scale=pin)
tr_ids = set(extract_gen_tac_film(file_path=[tr], scale=pin)[0].keys())
ids = [k for k in data if k not in tr_ids]
b = collate_fn_tacro_film([TacroFilmDataset({k: data[k] for k in ids})[i] for i in range(len(ids))], dev)
adv = b['static_v1'][:, 1] < 0.5
TG = torch.linspace(0.0, 24.0, 97)                                                      # 0.25-h grid
TAUS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 12.0, 18.0]
with torch.no_grad():
    x, st = model._encoder_input(b['observed_data_v1'], b['dose_v1'], b['static_v1'])
    mu, _ = model.encoder_z0(x, b['observed_tp_v1'], static=st, run_backwards=True)    # [1, N, L]
    z0 = mu[:, adv]; d1, d2 = b['dose_v1'][adv], b['dose_v2'][adv]
    c = model._film_context(torch.stack([d1, d2], 1).unsqueeze(0), z0.detach(), None)
    gam, bet = model._film_maps(c, d1, d2, 1)
    conc = lambda traj: unscale_data(model.decoder(traj).squeeze(-1), pin)[0]             # [N, T] ng/mL
    tr_old = model.diffeq_solver(z0, TG)                                                 # [1, N, T, L]
    tr_new0 = model.diffeq_solver(model._film_apply(z0, gam, bet), TG)
    c_old, c_new0 = conc(tr_old), conc(tr_new0)
    tg = TG.numpy(); ratios = {}; curves = {}
    for tau in TAUS:
        k = int(np.argmin(np.abs(tg - tau)))
        z_t = tr_old[:, :, k]
        tr_t = model.diffeq_solver(model._film_apply(z_t, gam, bet), TG[k:])
        c_t = conc(tr_t)
        a_t = np.trapezoid(c_t, tg[k:], axis=1); a_0 = np.trapezoid(c_new0[:, k:], tg[k:], axis=1)
        ratios[tau] = a_t / a_0
        curves[tau] = c_t
check0 = float(np.abs(curves[0.0] - c_new0).max())
dmax = float(pin[2]) if len(pin) > 2 else 8.0
r_dose = (d2 / d1).numpy()
i_ex = int(np.argmin(np.abs(r_dose - 2.0) + 0.001 * np.abs(np.median(c_old, 0) - c_old).mean(1) / np.median(c_old)))
med = np.array([np.median(np.abs(ratios[t] - 1)) for t in TAUS]) * 100
q10 = np.array([np.quantile(ratios[t], 0.10) for t in TAUS]); q90 = np.array([np.quantile(ratios[t], 0.90) for t in TAUS])
q50 = np.array([np.median(ratios[t]) for t in TAUS])
np.savez_compressed(OUTP + '.npz', tg=tg, taus=np.array(TAUS), med_abs_pct=med, q10=q10, q50=q50, q90=q90,
                    ex_old=c_old[i_ex], ex_new0=c_new0[i_ex], ex_d1=float(d1[i_ex]) * dmax, ex_d2=float(d2[i_ex]) * dmax,
                    **{f'ex_tau{t:g}': curves[t][i_ex] for t in TAUS})
log = [f'OT-FiLM {ck}', f'Advagraf test patients {int(adv.sum())}; CHECK transport at t=0 vs the model\'s own prediction: max |diff| {check0:.2e} ng/mL',
       'AUC[t, 24 h] with the transport applied at t, relative to the transport at the dose (then the flow):',
       '  t (h)          ' + ' '.join(f'{t:6g}' for t in TAUS),
       '  median |r-1| % ' + ' '.join(f'{v:6.1f}' for v in med),
       '  median r       ' + ' '.join(f'{v:6.2f}' for v in q50),
       '  10 % / 90 %    ' + ' '.join(f'{lo:.2f}/{hi:.2f}' for lo, hi in zip(q10, q90)),
       f'example patient: d1 {float(d1[i_ex]) * dmax:.0f} mg -> d2 {float(d2[i_ex]) * dmax:.0f} mg']
open(OUTP + '.txt', 'w').write('\n'.join(log) + '\n')
print('\n'.join(log))
