#!/usr/bin/env python3
"""B VERSION 2 on Baaz et al. 2026's own regimen test. Same data, patient check, design and metrics as dh_regimen.py (B as is, which the running queue still
calls -- left untouched, so this is a separate file); only the model changes, as diagnosed on the first B runs:
  1. FIXED observation noise (--obsrv-std, default 0.05 in units of the max training observation = the main config's
     value, not tuned to their true noise). B as is LEARNED it and, with 6 samples per patient, inflated it (54 mg/L vs a
     true 10) until the individual latent collapsed (KL ~0.01) and it predicted the population curve.
  2. Drug dynamics and readout LINEAR in the drug state, rates set by physiology. With the learned orthonormal basis Q
     (--orth-split 1,5: drug subspace u = Q_d' z, 6 dims; physiology p = Q_p' z, 4 dims, conserved):
        du/dt = A(p) u,  A(p) compartmental: A_ij = F_ij(p) >= 0 (transfer j -> i), A_jj = -(sum_i F_ij + e_j(p)), e > 0
                 -> Hurwitz by construction (no unstable drug-free equilibrium, which B as is developed);
        concentration = c(p)' u with c >= 0 -> no drug gives EXACTLY zero (B as is had a phantom exposure);
        dose = u -> u + (amount / 800 mg) * b, b >= 0 (amounts enter compartments).
     Between doses the system is linear with constant coefficients, so it is solved EXACTLY by matrix exponentials (no ODE
     solver). Exact superposition in dose: suits their linear simulator; would NOT capture saturable (Michaelis-Menten)
     elimination as is -- the trade-offaccepted.
  The initial drug state is encoded (their patients start with drug on board), as in dh_regimen.py.
    dh_regimen_v2.py <n_per_arm> <dense|sparse> <seed> <out.json> [--iters 6000] [--obsrv-std 0.05] [--latents 10] [--split 1,5]"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, time, argparse, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.getcwd())
sys.path.insert(0, f'{REPO}/third_party/baaz_ebvae/ours/regimen')
import gen_regimens as G
from probe_entanglement import _priors
from lib.create_latent_ode_model import create_LatentODE_model

ap = argparse.ArgumentParser()
ap.add_argument('n_per_arm', type=int); ap.add_argument('design', choices=['dense', 'sparse'])
ap.add_argument('seed', type=int); ap.add_argument('out')
ap.add_argument('--iters', type=int, default=6000)
ap.add_argument('--obsrv-std', type=float, default=0.05)
ap.add_argument('--latents', type=int, default=10)
ap.add_argument('--split', default='1,5')
ap.add_argument('--no-check', action='store_true', help='skip the comparison with the EB-VAE run CSVs')
ap.add_argument('--protocol', choices=['cf', 'firstint'], default='cf',
                help='firstint = the iScience format: the encoder sees only the first dosing interval, the model predicts '
                     'the rest of the SAME regimen; test patients observed under every regimen (stream seed + 4242); '
                     'scored on [second dose, 30 h]')
ap.add_argument('--enc-dose', choices=['raw', 'norm', 'none'], default='raw',
                help='what the ENCODER is told about the dose (dynamics always get the true doses): raw = concentrations + '
                     'cumulative dose + arm dose (default); norm = concentrations divided by the dose (relative to 800 mg), '
                     'dose count only, no dose value; none = concentrations + dose count only')
ap.add_argument('--titrate', type=float, default=None,
                help='titrated training arms (gen_regimens.titrate: fast eliminators get 800 mg; own random stream, so '
                     'patients and noise are those of the randomised run); test arms unchanged')
cli = ap.parse_args()
NA, DESIGN, seed = cli.n_per_arm, cli.design, cli.seed
torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
dev = torch.device('cpu'); t_start = time.time()
SAMP = np.arange(0, 30.25, 0.5) if DESIGN == 'dense' else np.array([0, 2, 7.5, 15.5, 23.5, 28.0])
EG = np.arange(0, 30.0001, 0.25)
TRAIN_TIMES = [0, 8, 16, 24]

# ---------------------------------------------------------------- data: run_regimen.py's exact sequence (as dh_regimen.py)
def sample_obs(P, times, amounts, rng):
    t_eval, c = G.simulate(P, times, amounts)
    return t_eval, c, G.noisy(np.stack([np.interp(SAMP, t_eval, ci) for ci in c]), rng)

rng = np.random.default_rng(seed)
n_tr = 2 * NA; P_tr = G.draw(n_tr, rng)
if cli.titrate is None:
    arm_tr, titr_corr = np.where(np.arange(n_tr) % 2 == 0, 400.0, 800.0), None
else:
    arm_tr, titr_corr = G.titrate(P_tr, cli.titrate, seed)
    print(f'TITRATED training arms: rho {cli.titrate}, corr(log ke, 800 mg) {titr_corr:+.2f}', flush=True)
A_tr = np.repeat(arm_tr[:, None], 4, 1); _, _, y_tr = sample_obs(P_tr, TRAIN_TIMES, A_tr, rng)
n_te = 400; P_te = G.draw(n_te, rng); arm_te = np.where(np.arange(n_te) % 2 == 0, 400.0, 800.0)
A_te = np.repeat(arm_te[:, None], 4, 1); _, _, y_te = sample_obs(P_te, TRAIN_TIMES, A_te, rng)
truth, regs = {}, {}
for name, (times, amts) in G.REGIMENS.items():
    A = A_te if amts is None else np.tile(np.array(amts, float), (n_te, 1)); regs[name] = (list(times), A)
    t_eval, truth[name] = G.simulate(P_te, times, A)
if not cli.no_check and cli.titrate is None:   # same patients as the EB-VAE run of this configuration?
    import pandas as pd
    tag = f'N{NA}_{DESIGN}' + (f'_s{seed}' if seed else '')
    wd = f'{REPO}/third_party/baaz_ebvae/ours/regimen/res/{tag}_data'
    for f, y in (('train.csv', y_tr), ('test_factual.csv', y_te)):
        d = pd.read_csv(f'{wd}/{f}', sep=';'); d = d[d.EVID == 0].sort_values(['ID', 'TIME'])
        ok = np.allclose(d.DV.values, y.reshape(-1), rtol=0, atol=1e-6)
        print(f'data check vs {tag}/{f}: {"identical" if ok else "MISMATCH"}', flush=True)
        if not ok:
            sys.exit('data differ from the EB-VAE run -- refusing to continue')

FI = cli.protocol == 'firstint'
def t2_of(times):
    return float(sorted(times)[1])          # second dose = end of the first dosing interval (8 h, or 3 h for time400/800)
if FI:   # every test patient observed under every regimen, own stream so the main data stream is untouched
    rng_r = np.random.default_rng(seed + 4242)
    y_obs = {name: G.noisy(np.stack([np.interp(SAMP, t_eval, c) for c in truth[name]]), rng_r) for name in G.REGIMENS}

YMAX = float(y_tr.max()); DMAX = 800.0
print(f'{n_tr} train patients, {len(SAMP)} samples each, YMAX {YMAX:.1f}, test {n_te}', flush=True)

def tens(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)

def enc_inputs(y, amounts, times):
    """[y / YMAX, cumulative dose so far / (4 * DMAX)] at the sample times; static = [arm dose / DMAX]"""
    cum = np.stack([(amounts * (np.array(times)[None, :] <= t)).sum(1) for t in SAMP], 1) / (4 * DMAX)
    if cli.enc_dose == 'raw':
        data = np.stack([y / YMAX, cum], -1)
        return tens(data), tens(amounts[:, :1] / DMAX)
    count = np.stack([(np.array(times)[None, :] <= t).sum(1).repeat(len(y)) for t in SAMP], 1) / 4.0   # doses so far, no amount
    scale = amounts[:, :1] / DMAX if cli.enc_dose == 'norm' else np.ones((len(y), 1))
    return tens(np.stack([y / YMAX / scale, count], -1)), tens(np.ones((len(y), 1)))

X_tr, S_tr = enc_inputs(y_tr, A_tr, TRAIN_TIMES)
X_te, S_te = enc_inputs(y_te, A_te, TRAIN_TIMES)
Y_tr = tens(y_tr / YMAX)
tp_samp = tens(SAMP)


M_TR = torch.as_tensor(SAMP < t2_of(TRAIN_TIMES)) if FI else None
X_enc_tr, tp_enc_tr = (X_tr[:, M_TR], tp_samp[M_TR]) if FI else (X_tr, tp_samp)


def fi_eval(res, encoder, predict_fn):
    """first-interval protocol: encode the first dosing interval of each regimen, predict that regimen, score [t2, 30 h]"""
    res['protocol'] = 'firstint'; res['window_start'] = {}; res.pop('population', None)
    with torch.no_grad():
        for name in G.REGIMENS:
            times, A = regs[name]; cut = t2_of(times); m = torch.as_tensor(SAMP < cut)
            X_r, S_r = enc_inputs(y_obs[name], A, times)
            mu, _ = encoder(X_r[:, m], tp_samp[m], static=S_r, run_backwards=True)
            w = EG >= cut - 1e-9
            pred = predict_fn(mu, times, A)[:, w]
            tru = np.stack([np.interp(EG, t_eval, c) for c in truth[name]])[:, w]
            auc_p, auc_t = np.trapezoid(pred, EG[w], axis=1), np.trapezoid(tru, EG[w], axis=1)
            r = auc_p / auc_t - 1; nrmse = np.sqrt(np.mean((pred - tru) ** 2, 1)) / np.mean(tru, 1)
            res['individual'][name] = dict(auc_abs_median=100 * float(np.median(np.abs(r))), nrmse_median=100 * float(np.median(nrmse)),
                                           auc_bias_median=100 * float(np.median(r)), auc_err_per_patient=[float(x) for x in r],
                                           auc_pred_per_patient=[float(x) for x in auc_p])
            res['window_start'][name] = cut
            print(f'{name:8s} first {cut:g} h observed -> |AUC({cut:g}-30 h) err| {res["individual"][name]["auc_abs_median"]:5.1f}%  '
                  f'nRMSE {res["individual"][name]["nrmse_median"]:5.1f}%  bias {res["individual"][name]["auc_bias_median"]:+5.1f}%', flush=True)

# ---------------------------------------------------------------- model: B's encoder + learned basis, linear compartments
T = 'confound_vc00_s4_pnoise_win_n100'
import argparse as _argparse, json as _json
args = _argparse.Namespace(**_json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dose_event_base_args.json'))))   # architecture flags of the model
args.use_film = False; args.latents = cli.latents; args.static_dim = 1
args.orth_split = cli.split; args.dose_history = 0                                      # basis only; no flow is used
obsrv_std, z0_prior = _priors(dev, 0.01)
base = create_LatentODE_model(args, 1, z0_prior, obsrv_std, dev)
encoder, osplit = base.encoder_z0, base.orth_split
K = osplit.k_f + osplit.k_d; KP = cli.latents - K


class LinComp(nn.Module):
    """physiology p -> compartmental rate matrix A(p) (Hurwitz), readout c(p) >= 0; dose vector b >= 0"""
    def __init__(self, kp, k, hidden=32):
        super().__init__()
        self.k = k
        self.rates = nn.Sequential(nn.Linear(kp, hidden), nn.Tanh(), nn.Linear(hidden, k * k + k))
        self.readout = nn.Sequential(nn.Linear(kp, hidden), nn.Tanh(), nn.Linear(hidden, k))
        self.kick_raw = nn.Parameter(torch.zeros(k))
        with torch.no_grad():   # start: weak transfers (~0.05 /h), elimination ~0.1 /h, equal readout, physiology-free
            self.rates[-1].weight.zero_(); self.rates[-1].bias[:k * k].fill_(-3.0); self.rates[-1].bias[k * k:].fill_(-2.25)
            self.readout[-1].weight.zero_(); self.readout[-1].bias.zero_()

    def A(self, p):
        raw = self.rates(p); k = self.k
        eye = torch.eye(k, device=p.device, dtype=p.dtype)
        Fm = F.softplus(raw[..., :k * k]).reshape(*p.shape[:-1], k, k) * (1 - eye)   # Fm[i, j] = transfer j -> i
        e = F.softplus(raw[..., k * k:])
        return Fm - torch.diag_embed(Fm.sum(-2) + e)

    def c(self, p):
        return F.softplus(self.readout(p))

    def b(self):
        return F.softplus(self.kick_raw)


lin = LinComp(KP, K).to(dev)


def predict(z0, dose_times, amounts, t_out):
    """z0 [S,B,D] = latent state at t=0 before any dose; dose_times shared (sorted); amounts [B,K_doses] mg; t_out sorted.
    Normalised concentration [S,B,len(t_out)]. A dose at t applies before the output at t (their simulator)."""
    Q = osplit.Q()
    u, p = z0 @ Q[:, :K], z0 @ Q[:, K:]
    Am, c, b = lin.A(p), lin.c(p), lin.b()
    a = tens(amounts / DMAX)
    t_out = [float(t) for t in t_out]
    outs = {}

    def advance(u, ta, tb, seg):
        """exact solution u(t) = expm(A (t - ta)) u(ta) at the outputs in seg and at tb; returns u(tb)"""
        dts = [t - ta for t in seg] + [tb - ta]
        if max(dts) <= 0:
            for t in seg:
                outs[t] = u
            return u
        E = torch.linalg.matrix_exp(Am.unsqueeze(2) * tens(dts).reshape(1, 1, -1, 1, 1))      # [S,B,n,K,K]
        U = (E @ u.unsqueeze(2).unsqueeze(-1)).squeeze(-1)                                    # [S,B,n,K]
        for j, t in enumerate(seg):
            outs[t] = U[:, :, j]
        return U[:, :, -1]

    tc = 0.0
    for kd, t_k in enumerate(float(t) for t in dose_times):
        if t_k > t_out[-1]:
            break
        u = advance(u, tc, t_k, [t for t in t_out if tc <= t < t_k])
        u = u + a[:, kd].reshape(1, -1, 1) * b
        tc = t_k
    advance(u, tc, t_out[-1], [t for t in t_out if t >= tc])
    U = torch.stack([outs[t] for t in t_out], 2)                                               # [S,B,T,K]
    return (U * c.unsqueeze(2)).sum(-1)


def main():
    params = list(encoder.parameters()) + list(osplit.parameters()) + list(lin.parameters())
    opt = torch.optim.Adamax(params, lr=0.01)
    S, sd = 3, cli.obsrv_std
    print(f'model: B version 2 (orth-split {cli.split}: drug {K} / physiology {KP}; linear compartments, exact solution; '
          f'fixed noise sd {sd} = {sd * YMAX:.1f} mg/L), {sum(p.numel() for p in params)} params, {cli.iters} iterations', flush=True)
    for it in range(1, cli.iters + 1):
        for g in opt.param_groups:
            g['lr'] = max(g['lr'] * 0.999, 0.001)
        kl_coef = 0.0 if it < 20 else 1 - 0.99 ** (it - 20)
        opt.zero_grad()
        mu, std = encoder(X_enc_tr, tp_enc_tr, static=S_tr, run_backwards=True)
        std = std.abs().clamp(min=1e-5)
        z0 = mu + std * torch.randn(S, *mu.shape[1:])
        pred = predict(z0, TRAIN_TIMES, A_tr, SAMP)
        loglik = torch.distributions.Normal(pred, sd).log_prob(Y_tr.unsqueeze(0)).sum(-1)           # [S,B]
        kl = torch.distributions.kl_divergence(torch.distributions.Normal(mu, std), torch.distributions.Normal(0., 1.)).sum(-1)
        iwae = torch.logsumexp(loglik - kl_coef * kl, 0) - np.log(S)
        loss = -iwae.mean() / len(SAMP)
        if not torch.isfinite(loss):
            print(f'iteration {it}: non-finite loss, stopping', flush=True); break
        loss.backward(); opt.step()
        if it % 250 == 0 or it == 1:
            with torch.no_grad():
                rmse = ((pred.mean(0) - Y_tr) ** 2).mean().sqrt().item() * YMAX
            print(f'it {it:5d}  loss {loss.item():9.4f}  kl {kl.mean().item():7.3f}  train RMSE {rmse:7.2f} mg/L  '
                  f'({(time.time() - t_start) / 60:.1f} min)', flush=True)
    minutes_train = (time.time() - t_start) / 60

    res = dict(model='B2-linear-compartments', N_per_arm=NA, design=DESIGN, seed=seed, n_samples_per_patient=len(SAMP),
               iters=cli.iters, latents=cli.latents, split=cli.split, obsrv_std=cli.obsrv_std, enc_dose=cli.enc_dose, minutes_train=minutes_train, titrate=cli.titrate, titrate_corr=titr_corr,
               individual={}, population={})
    encoder.eval()
    if FI:
        fi_eval(res, encoder, lambda mu, times, A: predict(mu, times, A, EG)[0].numpy() * YMAX)
    else:
     with torch.no_grad():
        mu_te, _ = encoder(X_te, tp_samp, static=S_te, run_backwards=True)
        Q = osplit.Q(); ev = torch.linalg.eigvals(lin.A(mu_te @ Q[:, K:])).real
        res['max_re_eig_test'] = float(ev.max())
        P_pop = G.draw(2000, np.random.default_rng(seed + 777))
        for name in G.REGIMENS:
            times, A = regs[name]
            pred = predict(mu_te, times, A, EG)[0].numpy() * YMAX
            tru = np.stack([np.interp(EG, t_eval, c) for c in truth[name]])
            r = np.trapezoid(pred, EG, axis=1) / np.trapezoid(tru, EG, axis=1) - 1
            nrmse = np.sqrt(np.mean((pred - tru) ** 2, 1)) / np.mean(tru, 1)
            res['individual'][name] = dict(auc_abs_median=100 * float(np.median(np.abs(r))), nrmse_median=100 * float(np.median(nrmse)),
                                           auc_bias_median=100 * float(np.median(r)), auc_err_per_patient=[float(x) for x in r])
            A_pop = (np.repeat(np.where(np.arange(2000) % 2 == 0, 400.0, 800.0)[:, None], len(times), 1) if name == 'train'
                     else np.tile(A[:1], (2000, 1)))
            _, c_pop = G.simulate(P_pop, times, A_pop)
            q_true = np.percentile(np.stack([np.interp(EG, t_eval, c) for c in c_pop]), [10, 50, 90], 0)
            sims = predict(torch.randn(1, 2000, cli.latents), times, A_pop, EG)[0].numpy() * YMAX
            q_pred = np.percentile(sims, [10, 50, 90], 0)
            res['population'][name] = {f'q{q}_mae_rel': 100 * float(np.mean(np.abs(q_pred[j] - q_true[j])) / np.mean(q_true[1]))
                                       for j, q in enumerate((10, 50, 90))}
            print(f'{name:8s} individual |AUC err| {res["individual"][name]["auc_abs_median"]:5.1f}%  nRMSE '
                  f'{res["individual"][name]["nrmse_median"]:5.1f}%  bias {res["individual"][name]["auc_bias_median"]:+5.1f}%  | population '
                  f'q10/q50/q90 ' + ' / '.join(f'{v:.1f}%' for v in res['population'][name].values()), flush=True)
    if 'max_re_eig_test' in res:
        print(f'max Re eigenvalue of A(p) over test patients: {res["max_re_eig_test"]:+.4f} (< 0 by construction)', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    json.dump(res, open(cli.out, 'w'), indent=1)
    torch.save(dict(encoder=encoder.state_dict(), osplit=osplit.state_dict(), lin=lin.state_dict(), args=args, cli=vars(cli),
                    YMAX=YMAX), os.path.splitext(cli.out)[0] + '.pt')
    print(f'REGIMEN_DH_DONE {(time.time() - t_start) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
