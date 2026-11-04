#!/usr/bin/env python3
"""IMPROVED B. Built on dh_regimen_v2.py (B2: same data,
encoder, learned orthonormal basis drug 6 / physiology 4, dose = jump into the drug state, protocols cf / firstint, --titrate,
--enc-dose), left untouched because a queue runs it. Three opt-in changes, each its own switch (so they double as ablation):
  --resid-dyn LAMBDA   drug dynamics du/dt = A(p) u + [g(u, p) - g(0, p)]: B2's linear compartments (transfers >= 0,
                       elimination > 0) + a nonlinear correction that is ZERO with no drug, starts at zero (last layer
                       zero-initialised) and is penalised by LAMBDA * mean(correction^2) along the training trajectories.
                       Solved numerically (dopri5) between dose events. Off (default, -1): B2's exact linear solution.
  --resid-read LAMBDA  readout c(p)' u + [h(u, p) - h(0, p)], same rules (no drug -> exactly zero concentration).
  --resid-warmup N     corrections frozen at zero for N iterations (default 1000): the linear model fits first, the
                       nonlinear part only explains what it cannot (B, nonlinear from the start, absorbed the confounding).
  --noise learned      observation sd learned, but frozen at --obsrv-std for --noise-warmup iterations and floored at
                       --noise-floor (B learned it from the start with sparse data, inflated it to 54 mg/L and its latent
                       collapsed). Default fixed (= B2).
Validation (to choose LAMBDA WITHOUT the counterfactual truth): --val-n fresh patients (own stream seed + 5555) under the
training regimen and design; their reconstruction RMSE and Gaussian NLL are stored. Saturable simulator: REGIMEN_KM env
(gen_regimens.py); the EB-VAE CSV check is skipped automatically then.
    dh_regimen_v3.py <n_per_arm> <dense|sparse> <seed> <out.json> [--resid-dyn L] [--resid-read L] [--noise fixed|learned] ..."""
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
ap.add_argument('--resid-dyn', type=float, default=-1.0, help='penalty on the nonlinear dynamics correction (-1 = off)')
ap.add_argument('--resid-read', type=float, default=-1.0, help='penalty on the nonlinear readout correction (-1 = off)')
ap.add_argument('--resid-hidden', type=int, default=32)
ap.add_argument('--resid-warmup', type=int, default=1000, help='iterations with the corrections frozen at zero (linear model fits first)')
ap.add_argument('--noise', choices=['fixed', 'learned'], default='fixed')
ap.add_argument('--noise-warmup', type=int, default=1000)
ap.add_argument('--noise-floor', type=float, default=0.005)
ap.add_argument('--val-n', type=int, default=100)
ap.add_argument('--arm-balance', type=float, default=0.0,
                help='soft penalty tying the distribution of the encoded PHYSIOLOGY p (not the drug state) across the two training '
                     'arms, for randomised trials: w * (sum of squared standardised mean differences + squared Frobenius distance of the '
                     'standardised covariances), encoder means, pooled within-arm sd; 0 = off')
ap.add_argument('--load', default=None, help='evaluate a saved .pt (use with --iters 0 and the run\'s own flags; output to a NEW path)')
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
if not cli.no_check and cli.titrate is None and G.KM is None:   # same patients as the EB-VAE run of this configuration?
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
    """physiology p -> compartmental rate matrix A(p) (Hurwitz), readout c(p) >= 0; dose vector b >= 0 (as B2)"""
    def __init__(self, kp, k, hidden=32):
        super().__init__()
        self.k = k
        self.rates = nn.Sequential(nn.Linear(kp, hidden), nn.Tanh(), nn.Linear(hidden, k * k + k))
        self.readout = nn.Sequential(nn.Linear(kp, hidden), nn.Tanh(), nn.Linear(hidden, k))
        self.kick_raw = nn.Parameter(torch.zeros(k))
        with torch.no_grad():
            self.rates[-1].weight.zero_(); self.rates[-1].bias[:k * k].fill_(-3.0); self.rates[-1].bias[k * k:].fill_(-2.25)
            self.readout[-1].weight.zero_(); self.readout[-1].bias.zero_()

    def A(self, p):
        raw = self.rates(p); k = self.k
        eye = torch.eye(k, device=p.device, dtype=p.dtype)
        Fm = F.softplus(raw[..., :k * k]).reshape(*p.shape[:-1], k, k) * (1 - eye)
        e = F.softplus(raw[..., k * k:])
        return Fm - torch.diag_embed(Fm.sum(-2) + e)

    def c(self, p):
        return F.softplus(self.readout(p))

    def b(self):
        return F.softplus(self.kick_raw)


class Resid(nn.Module):
    """g(u, p) - g(0, p): exactly zero with no drug; last layer zero-initialised, so training starts at the linear model"""
    def __init__(self, k, kp, k_out, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k + kp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, k_out))
        with torch.no_grad():
            self.net[-1].weight.zero_(); self.net[-1].bias.zero_()

    def forward(self, u, p):
        return self.net(torch.cat([u, p], -1)) - self.net(torch.cat([torch.zeros_like(u), p], -1))


lin = LinComp(KP, K).to(dev)
_STATE = {'it': 0}
def rdyn_active():
    """the dynamics correction is exactly zero during the warm-up (frozen, zero last layer): use the exact solution then"""
    return _STATE['it'] == 0 or _STATE['it'] > cli.resid_warmup
rdyn = Resid(K, KP, K, cli.resid_hidden).to(dev) if cli.resid_dyn >= 0 else None
rread = Resid(K, KP, 1, cli.resid_hidden).to(dev) if cli.resid_read >= 0 else None
log_sd = nn.Parameter(torch.tensor(float(np.log(cli.obsrv_std))), requires_grad=False)


def latent_parts(z0):
    Q = osplit.Q()
    return z0 @ Q[:, :K], z0 @ Q[:, K:]


def predict(z0, dose_times, amounts, t_out, states=False):
    """z0 [S,B,D] = latent state at t=0 before any dose; dose_times shared (sorted); amounts [B,K_doses] mg; t_out sorted.
    Normalised concentration [S,B,T] (+ drug states [S,B,T,K] and physiology). A dose at t applies before the output at t."""
    from torchdiffeq import odeint
    u, p = latent_parts(z0)
    Am, c, b = lin.A(p), lin.c(p), lin.b()
    a = tens(amounts / DMAX)
    t_out = [float(t) for t in t_out]
    outs = {}

    def field(t, uu):
        return (Am @ uu.unsqueeze(-1)).squeeze(-1) + rdyn(uu, p)

    def advance(u, ta, tb, seg):
        dts = [t - ta for t in seg] + [tb - ta]
        if max(dts) <= 0:
            for t in seg:
                outs[t] = u
            return u
        if rdyn is None or not rdyn_active():   # B2's exact solution (also while the correction is frozen at zero)
            E = torch.linalg.matrix_exp(Am.unsqueeze(2) * tens(dts).reshape(1, 1, -1, 1, 1))
            U = (E @ u.unsqueeze(2).unsqueeze(-1)).squeeze(-1)
            for j, t in enumerate(seg):
                outs[t] = U[:, :, j]
            return U[:, :, -1]
        grid = sorted(set([ta] + seg + [tb]))
        sol = odeint(field, u, torch.tensor(grid, dtype=torch.float32), rtol=1e-3, atol=1e-5)
        for t in seg:
            outs[t] = sol[grid.index(t)]
        return sol[-1]

    tc = 0.0
    for kd, t_k in enumerate(float(t) for t in dose_times):
        if t_k > t_out[-1]:
            break
        u = advance(u, tc, t_k, [t for t in t_out if tc <= t < t_k])
        u = u + a[:, kd].reshape(1, -1, 1) * b
        tc = t_k
    advance(u, tc, t_out[-1], [t for t in t_out if t >= tc])
    U = torch.stack([outs[t] for t in t_out], 2)                                               # [S,B,T,K]
    y = (U * c.unsqueeze(2)).sum(-1)
    if rread is not None:
        y = y + rread(U, p.unsqueeze(2).expand(*U.shape[:-1], p.shape[-1])).squeeze(-1)
    return (y, U, p) if states else y


def penalty(U, p):
    """mean squared nonlinear corrections at the trajectory states (normalised units); returns (total, parts)"""
    pp = p.unsqueeze(2).expand(*U.shape[:-1], p.shape[-1])
    parts = {}
    if rdyn is not None:
        parts['dyn'] = (rdyn(U, pp) ** 2).mean()
    if rread is not None:
        parts['read'] = (rread(U, pp) ** 2).mean()
    tot = sum((max(cli.resid_dyn, 0) if k == 'dyn' else max(cli.resid_read, 0)) * v for k, v in parts.items()) if parts else torch.zeros(())
    return tot, parts


def arm_balance(mu, is800):
    """arm difference of the encoded physiology p (encoder means): (sum_k d_k^2 with d = standardised mean difference,
    ||C_800 - C_400||_F^2 with C = covariance of p standardised by the pooled within-arm sd)"""
    pm = latent_parts(mu.reshape(-1, mu.shape[-1]))[1]
    a, b = pm[is800], pm[~is800]
    sd = torch.cat([a - a.mean(0), b - b.mean(0)]).std(0).clamp(min=1e-6)
    d = (a.mean(0) - b.mean(0)) / sd
    za, zb = (a - a.mean(0)) / sd, (b - b.mean(0)) / sd
    Ca, Cb = za.T @ za / (len(za) - 1), zb.T @ zb / (len(zb) - 1)
    return (d ** 2).sum(), ((Ca - Cb) ** 2).sum()


IS800_TR = torch.as_tensor(arm_tr == 800.0)


def noise_sd():
    return log_sd.exp().clamp(min=cli.noise_floor) if cli.noise == 'learned' else torch.tensor(cli.obsrv_std)


# ---------------------------------------------------------------- validation cohort (choose the penalty without the counterfactual truth)
rng_v = np.random.default_rng(seed + 5555)
P_val = G.draw(cli.val_n, rng_v); A_val = np.repeat(np.where(np.arange(cli.val_n) % 2 == 0, 400.0, 800.0)[:, None], 4, 1)
_, _, y_val = sample_obs(P_val, TRAIN_TIMES, A_val, rng_v)
X_val, S_val = enc_inputs(y_val, A_val, TRAIN_TIMES)
X_enc_val = X_val[:, M_TR] if FI else X_val


def main():
    params = list(encoder.parameters()) + list(osplit.parameters()) + list(lin.parameters())
    params += (list(rdyn.parameters()) if rdyn is not None else []) + (list(rread.parameters()) if rread is not None else [])
    opt = torch.optim.Adamax(params + [log_sd], lr=0.01)
    S = 3
    desc = (f'dynamics {"linear (exact)" if rdyn is None else f"linear + correction (penalty {cli.resid_dyn:g})"}, readout '
            f'{"linear" if rread is None else f"linear + correction (penalty {cli.resid_read:g})"}, noise {cli.noise} '
            f'(start {cli.obsrv_std} = {cli.obsrv_std * YMAX:.1f} mg/L), simulator {"linear" if G.KM is None else f"saturable Km {G.KM:g}"}')
    print(f'model: improved B ({desc}), {sum(p.numel() for p in params)} params, {cli.iters} iterations', flush=True)
    if cli.load:
        ck = torch.load(cli.load, map_location='cpu', weights_only=False)
        encoder.load_state_dict(ck['encoder']); osplit.load_state_dict(ck['osplit']); lin.load_state_dict(ck['lin'])
        if rdyn is not None:
            rdyn.load_state_dict(ck['rdyn'])
        if rread is not None:
            rread.load_state_dict(ck['rread'])
        with torch.no_grad():
            log_sd.copy_(ck['log_sd'])
        assert abs(ck['YMAX'] - YMAX) < 1e-6, 'saved model was trained on other data'
        print(f'loaded {cli.load}', flush=True)
    for it in range(1, cli.iters + 1):
        for g in opt.param_groups:
            g['lr'] = max(g['lr'] * 0.999, 0.001)
        if cli.noise == 'learned' and it == cli.noise_warmup + 1:
            log_sd.requires_grad_(True)
        if it == 1 or it == cli.resid_warmup + 1:   # corrections frozen (exactly zero) during the warm-up
            for m in (rdyn, rread):
                if m is not None:
                    m.requires_grad_(it > cli.resid_warmup)
        kl_coef = 0.0 if it < 20 else 1 - 0.99 ** (it - 20)
        _STATE['it'] = it
        opt.zero_grad()
        mu, std = encoder(X_enc_tr, tp_enc_tr, static=S_tr, run_backwards=True)
        std = std.abs().clamp(min=1e-5)
        z0 = mu + std * torch.randn(S, *mu.shape[1:])
        pred, U, p = predict(z0, TRAIN_TIMES, A_tr, SAMP, states=True)
        sd = noise_sd()
        loglik = torch.distributions.Normal(pred, sd).log_prob(Y_tr.unsqueeze(0)).sum(-1)
        kl = torch.distributions.kl_divergence(torch.distributions.Normal(mu, std), torch.distributions.Normal(0., 1.)).sum(-1)
        iwae = torch.logsumexp(loglik - kl_coef * kl, 0) - np.log(S)
        pen, parts = penalty(U, p)
        loss = -iwae.mean() / len(SAMP) + pen
        if cli.arm_balance > 0:
            bal_m, bal_c = arm_balance(mu, IS800_TR)
            loss = loss + cli.arm_balance * (bal_m + bal_c)
        if not torch.isfinite(loss):
            print(f'iteration {it}: non-finite loss, stopping', flush=True); break
        loss.backward(); opt.step()
        if it % 250 == 0 or it == 1:
            with torch.no_grad():
                rmse = ((pred.mean(0) - Y_tr) ** 2).mean().sqrt().item() * YMAX
            print(f'it {it:5d}  loss {loss.item():9.4f}  kl {kl.mean().item():7.3f}  noise {sd.item() * YMAX:6.2f} mg/L  train RMSE {rmse:7.2f}  '
                  + '  '.join(f'corr_{k} {v.item():.2e}' for k, v in parts.items())
                  + (f'  arm-balance p: mean {bal_m.item():.3f} cov {bal_c.item():.3f}' if cli.arm_balance > 0 else '')
                  + f'  ({(time.time() - t_start) / 60:.1f} min)', flush=True)
    minutes_train = (time.time() - t_start) / 60
    _STATE['it'] = 0   # evaluation: full model

    res = dict(model='improved-B', N_per_arm=NA, design=DESIGN, seed=seed, n_samples_per_patient=len(SAMP), iters=cli.iters,
               latents=cli.latents, split=cli.split, resid_dyn=cli.resid_dyn, resid_read=cli.resid_read, resid_warmup=cli.resid_warmup, noise=cli.noise,
               noise_sd_final=float(noise_sd() * YMAX), enc_dose=cli.enc_dose, simulator_km=G.KM, titrate=cli.titrate,
               titrate_corr=titr_corr, minutes_train=minutes_train, individual={})
    encoder.eval()
    with torch.no_grad():
        # validation (fresh patients, training regimen, same design): reconstruction from their own samples
        mu_v, _ = encoder(X_enc_val, tp_enc_tr, static=S_val, run_backwards=True)
        pv = predict(mu_v, TRAIN_TIMES, A_val, SAMP)[0]
        Yv = tens(y_val / YMAX)
        res['validation'] = dict(rmse_mgL=float(((pv - Yv) ** 2).mean().sqrt() * YMAX),
                                 nll_per_sample=float(-torch.distributions.Normal(pv, noise_sd()).log_prob(Yv).mean()))
        # how nonlinear did it become? correction size relative to the linear part, test patients, training regimen
        mu_te, _ = encoder(X_te, tp_samp, static=S_te, run_backwards=True)
        # arm difference of the encoded physiology (training arms; test patients by the arm they were observed on, randomised)
        bm_tr, bc_tr = arm_balance(encoder(X_enc_tr, tp_enc_tr, static=S_tr, run_backwards=True)[0], IS800_TR)
        bm_te, bc_te = arm_balance(mu_te, torch.as_tensor(A_te[:, 0] == 800.0))
        res['arm_balance'] = cli.arm_balance
        res['p_arm_difference'] = dict(train_mean_sq_d=float(bm_tr), train_cov_frob_sq=float(bc_tr),
                                       test_mean_sq_d=float(bm_te), test_cov_frob_sq=float(bc_te))
        print(f'physiology p, arm 800 vs 400 (sum of squared standardised mean differences / covariance distance): train '
              f'{float(bm_tr):.3f} / {float(bc_tr):.3f}, test {float(bm_te):.3f} / {float(bc_te):.3f}', flush=True)
        y_lin_te, U_te, p_te = predict(mu_te, TRAIN_TIMES, A_te, EG, states=True)
        pp = p_te.unsqueeze(2).expand(*U_te.shape[:-1], p_te.shape[-1])
        if rdyn is not None:
            lin_part = (lin.A(p_te).unsqueeze(2) @ U_te.unsqueeze(-1)).squeeze(-1)
            res['dyn_correction_rel'] = float(rdyn(U_te, pp).norm(dim=-1).mean() / lin_part.norm(dim=-1).mean())
        if rread is not None:
            res['read_correction_rel'] = float(rread(U_te, pp).abs().mean() / (U_te * lin.c(p_te).unsqueeze(2)).sum(-1).abs().mean())
        print(f'validation (fresh patients): RMSE {res["validation"]["rmse_mgL"]:.2f} mg/L, NLL/sample {res["validation"]["nll_per_sample"]:.3f}; '
              f'corrections: dyn {res.get("dyn_correction_rel", 0):.3f}, read {res.get("read_correction_rel", 0):.3f} of the linear part', flush=True)
    if FI:
        fi_eval(res, encoder, lambda mu, times, A: predict(mu, times, A, EG)[0].numpy() * YMAX)
    else:
        with torch.no_grad():
            for name in G.REGIMENS:
                times, A = regs[name]
                pred = predict(mu_te, times, A, EG)[0].numpy() * YMAX
                tru = np.stack([np.interp(EG, t_eval, c) for c in truth[name]])
                auc_p, auc_t = np.trapezoid(pred, EG, axis=1), np.trapezoid(tru, EG, axis=1)
                r = auc_p / auc_t - 1; nrmse = np.sqrt(np.mean((pred - tru) ** 2, 1)) / np.mean(tru, 1)
                res['individual'][name] = dict(auc_abs_median=100 * float(np.median(np.abs(r))), nrmse_median=100 * float(np.median(nrmse)),
                                               auc_bias_median=100 * float(np.median(r)), auc_err_per_patient=[float(x) for x in r],
                                               auc_pred_per_patient=[float(x) for x in auc_p])
                print(f'{name:8s} individual |AUC err| {res["individual"][name]["auc_abs_median"]:5.1f}%  nRMSE '
                      f'{res["individual"][name]["nrmse_median"]:5.1f}%  bias {res["individual"][name]["auc_bias_median"]:+5.1f}%', flush=True)
            if all(n in res['individual'] for n in ('amt350', 'amt850')):
                k = np.log(850 / 350); t350 = np.trapezoid(np.stack([np.interp(EG, t_eval, c) for c in truth['amt350']]), EG, axis=1)
                t850 = np.trapezoid(np.stack([np.interp(EG, t_eval, c) for c in truth['amt850']]), EG, axis=1)
                ep = np.log(np.array(res['individual']['amt850']['auc_pred_per_patient']) / np.array(res['individual']['amt350']['auc_pred_per_patient'])) / k
                res['elasticity'] = dict(pred_median=float(np.nanmedian(ep)), true_median=float(np.median(np.log(t850 / t350) / k)))
                print(f'dose elasticity 350 -> 850 mg: predicted {res["elasticity"]["pred_median"]:.2f}, true {res["elasticity"]["true_median"]:.2f}', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    json.dump(res, open(cli.out, 'w'), indent=1)
    torch.save(dict(encoder=encoder.state_dict(), osplit=osplit.state_dict(), lin=lin.state_dict(),
                    rdyn=rdyn.state_dict() if rdyn is not None else None, rread=rread.state_dict() if rread is not None else None,
                    log_sd=log_sd.detach(), args=args, cli=vars(cli), YMAX=YMAX), os.path.splitext(cli.out)[0] + '.pt')
    print(f'REGIMEN_DH_DONE {(time.time() - t_start) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
