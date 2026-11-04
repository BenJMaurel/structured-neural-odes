#!/usr/bin/env python3
"""Data for the step-3 figure of ndm.tex: lines 302-305 "not clear at all, we need a figure to explain that using
PCA"): why the dose transport cannot represent a dosing history, and what the dose-event model does instead.
Two-visit tacrolimus test cohort (windowed scenario 4, in-range, Advagraf patients so that one dosing interval = 24 h in both
models), seed-1 checkpoints at ep6000, posterior means from visit 1, no training:
  OT-FiLM   (results/s4_pnoise_win/film_n100_s1; the step-2 model; the geometry runs of used the same ckpts)
            latent trajectory z(t) over one 24-h interval from the encoded state; ridge probe of log Vc fitted on z(0) (5-fold CV)
            and applied to z(t) (fixed read-out), and refitted at each t
  dose-event (results/s4_doseh/dh7pgkp_osplit1-5_n100_s1; the step-3 model of Table 2)
            the visit's 7-dose history replayed from the drug-free state with the model's own operations (jump u + a b(p), anchored
            drug flow), every interval recorded: drug coordinates u(t) = Q_d^T z(t), physiology p = Q_p^T z (constant); CHECK: the
            last recorded interval equals model._dose_history_traj on the same grid; ridge probe of log Vc on p.
Writes results/ndm/fig_transport_history.npz (+ .txt with the checks and probe values)."""
import os, sys, glob
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film, set_static_hematocrit
from probe_entanglement import load_film_model

torch.set_num_threads(1); dev = torch.device('cpu')
B = 'confound_vc00_s4_pnoise_win'; T = B + '_n100'; R = './results/exp_film_run'
CK = {'film': os.environ.get('FIG_FILM_ROOT', 'results/s4_pnoise_win') + '/film_n100_s1/exp_film_run/{T}/traj/*_ep006000.ckpt',   # FIG_FILM_ROOT 22/10 opt-in (gamma > 0)
      'event': 'results/s4_doseh/dh7pgkp_osplit1-5_n100_s1/exp_film_run/{T}/traj/*_ep006000.ckpt'}
TGRID = torch.linspace(0.0, 24.0, 49)
OUT = 'results/ndm/fig_transport_history'
os.makedirs('results/ndm', exist_ok=True)
log = []


def ridge():
    return Pipeline([('s', StandardScaler()), ('m', RidgeCV(alphas=np.logspace(-3, 3, 13)))])


def batch_for(model_args):
    set_static_hematocrit(int(getattr(model_args, 'static_dim', 3)) >= 4)
    pin = extract_gen_tac_film(file_path=[f'{R}/{T}/virtual_cohort_film_train.csv'])[1]
    tr, te = (f'{R}/{B}/virtual_cohort_film_{x}.csv' for x in ('train', 'test'))
    data, _ = extract_gen_tac_film(file_path=[tr, te], scale=pin)
    tr_ids = set(extract_gen_tac_film(file_path=[tr], scale=pin)[0].keys())
    ids = [k for k in data if k not in tr_ids]
    b = collate_fn_tacro_film([TacroFilmDataset({k: data[k] for k in ids})[i] for i in range(len(ids))], dev)
    adv = (b['static_v1'][:, 1] < 0.5).numpy()                       # Advagraf: 24-h interval
    return b, np.array([int(i) for i in b['patient_ids']]), adv


def encode(model, b):
    with torch.no_grad():
        x, st = model._encoder_input(b['observed_data_v1'], b['dose_v1'], b['static_v1'])
        mu, _ = model.encoder_z0(x, b['observed_tp_v1'], static=st, run_backwards=True)
    return mu[0]


truth = pd.read_csv(f'{R}/{B}/confound_truth.csv').set_index('ID')
res = {}
# ------------------------------------------------------------------ OT-FiLM
ck = sorted(glob.glob(CK['film'].format(T=T)))[0]
a = torch.load(ck, map_location='cpu', weights_only=False)['args']
model, _ = load_film_model(ck, dev); model.eval()
b, ids, adv = batch_for(a)
z0 = encode(model, b)[adv]
with torch.no_grad():
    zt = model.diffeq_solver(z0.unsqueeze(0), TGRID).squeeze(0).numpy()        # [N, T, L]
y = np.log(truth.loc[ids[adv], 'confound_par'].values.astype(float))          # central volume (confound_param == 'Vc')
assert (truth.loc[ids[adv], 'confound_param'] == 'Vc').all()
fixed, refit = np.zeros(len(TGRID)), np.zeros(len(TGRID))
pred_fixed = np.zeros((len(TGRID), len(y))); pred0 = np.zeros(len(y))
for trn, tst in KFold(5, shuffle=True, random_state=0).split(zt[:, 0]):
    m = ridge().fit(zt[trn, 0], y[trn])
    for k in range(len(TGRID)):
        pred_fixed[k, tst] = m.predict(zt[tst, k])
for k in range(len(TGRID)):
    fixed[k] = r2_score(y, pred_fixed[k])
    p_k = np.zeros(len(y))
    for trn, tst in KFold(5, shuffle=True, random_state=0).split(zt[:, k]):
        p_k[tst] = ridge().fit(zt[trn, k], y[trn]).predict(zt[tst, k])
    refit[k] = r2_score(y, p_k)
spread = np.sqrt(np.trace(np.cov(zt[:, 0].T)))
drift = np.median(np.linalg.norm(zt - zt[:, :1], axis=2), axis=0)
res.update(film_zt=zt, film_y=y, film_fixed=fixed, film_refit=refit, film_spread=spread, film_drift=drift)
log.append(f'OT-FiLM {ck}\n  Advagraf test patients {adv.sum()}; spread of z(0) (sqrt tr cov) {spread:.2f}; median drift |z(t)-z(0)| at '
           f'6/12/24 h {drift[12]:.1f} / {drift[24]:.1f} / {drift[48]:.1f}')
log.append('  probe log Vc: fixed (fit at t=0) at 0/1/3/6/12/24 h ' + ' / '.join(f'{fixed[k]:.2f}' for k in (0, 2, 6, 12, 24, 48))
           + ';  refit ' + ' / '.join(f'{refit[k]:.2f}' for k in (0, 2, 6, 12, 24, 48)))

# ------------------------------------------------------------------ dose-event model
ck = sorted(glob.glob(CK['event'].format(T=T)))[0]
a = torch.load(ck, map_location='cpu', weights_only=False)['args']
model, _ = load_film_model(ck, dev); model.eval()
b, ids_e, adv_e = batch_for(a)
assert (ids_e == ids).all() and (adv_e == adv).all()
z0 = encode(model, b)[adv]
doses = b['dose_v1'][adv]
dh, ode, os_ = model.dose_hist, model.diffeq_solver.ode_func, model.orth_split
Q = os_.Q().detach(); kd = os_.k_f + os_.k_d
with torch.no_grad():
    z = (z0 @ os_.P_phys()).unsqueeze(0)
    Qf = os_.Q(); k = Qf.shape[1] - dh.kick_net[0].in_features
    kick = doses.reshape(1, -1, 1) * ((dh.kick_coef + dh.kick_net(z0.unsqueeze(0) @ Qf[:, k:])) @ Qf[:, :k].t())
    ode._anchor = ode.gradient_net(z); ode._tscale = None
    segs = []
    try:
        for _ in range(dh.n_doses):
            sol = model.diffeq_solver(z + kick, TGRID)                          # [1, N, T, L]
            segs.append(sol[0]); z = sol[:, :, -1, :]
    finally:
        ode._anchor = None; ode._tscale = None
    hist = torch.cat(segs, 1).numpy()      # [N, 7*49, L]: full intervals, boundary times repeated (pre-dose end, post-dose start)
    own = model._dose_history_traj(z0.unsqueeze(0), doses, TGRID, form=b['static_v1'][adv, 1])[0].numpy()
gap = float(np.abs(own - segs[-1].numpy()).max())
u = hist @ Q[:, :kd].numpy(); p = hist @ Q[:, kd:].numpy()
p_moves = float(np.abs(p - p[:, :1]).max())
pe = np.zeros(len(y))
for trn, tst in KFold(5, shuffle=True, random_state=0).split(p[:, 0]):
    pe[tst] = ridge().fit(p[trn, 0], y[trn]).predict(p[tst, 0])
r2p = r2_score(y, pe)
th = np.concatenate([24.0 * i + TGRID.numpy() for i in range(dh.n_doses)])
res.update(event_u=u, event_p=p[:, 0], event_t=th, event_r2p=r2p, event_dose_mg=doses.numpy() * 8.0)
log.append(f'dose-event {ck}\n  CHECK replay vs model._dose_history_traj (last interval): max |diff| {gap:.2e}; physiology '
           f'coordinates move by at most {p_moves:.2e} over the 7 doses; |u| at the end of each interval '
           + ' / '.join(f'{np.median(np.linalg.norm(u[:, 49 * (i + 1) - 1], axis=1)):.2f}' for i in range(dh.n_doses))
           + f' (median), right after the first dose {np.median(np.linalg.norm(u[:, 0], axis=1)):.2f}')
log.append(f'  probe log Vc on p (5-fold CV): R2 {r2p:.2f} (same at every time: p is constant)')
np.savez_compressed(OUT + '.npz', tgrid=TGRID.numpy(), **res)
open(OUT + '.txt', 'w').write('\n'.join(log) + '\n')
print('\n'.join(log))
