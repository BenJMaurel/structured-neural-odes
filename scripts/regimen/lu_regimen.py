#!/usr/bin/env python3
"""Both Lu et al. 2021 designs on Baaz et al. 2026's regimen test (
our architecture is better than Lu's ... a test that would show that because of their model Lu confound things inside the
latent space"). Same data (run_regimen.py's random sequence, checked against the EB-VAE CSVs), same 400 test patients, same
regimens and metrics as dh_regimen.py / dh_regimen_v2.py.

  is  : iScience 2021 (Lu, Deng, Zhang, Liu & Guan), from the paper and their code (github jameslu01/Neural_PK):
        a GRU (128 units) scans REVERSELY the whole time series of the regimen being predicted, 5 channels per time point
        [TFDS (time since the last dose), TIME, CYCL (doses so far), AMT (amount dosed at that point), PK (the patient's
        observed concentrations at their sample times, 0 elsewhere = 'PK_cycle1 padded with zero')] -> 12 numbers = mean
        and sd of a 6-dim z0 (sampled in training, the mean at test; MSE loss, no KL, as their code); at each dosing time
        the amount is added to latent coordinate 0, then an autonomous ODE (4 fully connected layers of 16, SELU, weights
        N(0, 0.001), biases 0.5, as their code) runs to the next time point; decoder = [z_t, flattened observed (time, PK)
        vector] -> 32 SELU -> 1.
        Counterfactual, as their method: the encoder is fed the NEW regimen's dosing channels with the same observed PK, so
        z0 itself depends on the regimen being predicted ('target' protocol, reported as 'individual'). Ablation from the
        same trained model: encode with the FACTUAL dosing, change only the doses added during the rollout ('factual'
        protocol, 'individual_factual_encoder').
  nmi : NMI 2021 (Lu, Bender, Jin & Guan), from the preprint (arXiv 2010.11769) and lib/lu_neural_pk.py: deterministic GRU
        (state 20) over the observed data rows [time after dose, time, PK, amount dosed] -> 4 PK parameters; 2-dim PK state
        (dosed coordinate, concentration coordinate), forward Euler (step 0.25 h), the dose increments the dosed
        coordinate, PKVF([state, params]) = linear + SoftPlus layers (rates per hour), ReLU keeps the state non-negative, concentration =
        the second coordinate, L2 loss. Counterfactual = parameters frozen, new doses through the dose port.
        Deviation (as for our B): their InitState is zero ('no drug before dosing'), but Baaz's patients start with drug on
        board (central amount = v), so the encoder also emits the initial CONCENTRATION coordinate (softplus); the dosed
        coordinate starts empty, as theirs. Their field's output layer starts at zero (as our B), else it diverges.
Common deviations: full batch, Adam lr 1e-3 decayed 0.9995 / iteration to 1e-4, no weight decay (their lr 5e-5 / batch 1 /
L2 0.1 is a different optimisation regime), times / 30 h, concentrations / max training observation, doses / 800 mg.

Diagnostics (iScience only, 'identity'): test 1 of the plan -- the SAME patient with the SAME observations, encoded with each
regimen's dosing channels: identity ratio E||z0(r) - z0(train)||^2 / (2 tr Cov z0(train)) (0 = the regimen does not move the
patient, 1 = as different as two random patients), and a ridge probe of log ka / ke / v fitted on z0(train) and applied to
z0(r). Every model stores the predicted AUC(0-30 h) per patient and regimen (dose elasticity, scripts/regimen/lu_table.py).
--titrate RHO: test 2 -- training arms from gen_regimens.titrate (fast eliminators get 800 mg; own random stream, so the
patients and noise draws are those of the randomised runs); the test patients keep the alternating arms.

    lu_regimen.py <is|nmi> <n_per_arm> <dense|sparse> <seed> <out.json> [--iters 6000] [--titrate RHO] [--no-check]"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchdiffeq import odeint
sys.path.insert(0, f'{REPO}/third_party/baaz_ebvae/ours/regimen')
import gen_regimens as G

ap = argparse.ArgumentParser()
ap.add_argument('model', choices=['is', 'nmi'])
ap.add_argument('n_per_arm', type=int); ap.add_argument('design', choices=['dense', 'sparse', 'visit', 'visit1'])   # visit: --sim tacro only (NMI: off its 0.25-h grid)
ap.add_argument('seed', type=int); ap.add_argument('out')
ap.add_argument('--iters', type=int, default=6000)
ap.add_argument('--titrate', type=float, default=None)
ap.add_argument('--lr', type=float, default=1e-3, help='initial Adam learning rate (decayed 0.9995 / iteration to lr / 10)')
ap.add_argument('--hidden', type=int, default=32, help='NMI: width of the PKVF layers')
ap.add_argument('--wd', type=float, default=0.0, help='Adam weight decay (their L2; 0.1 in the iScience code)')
ap.add_argument('--batch', type=int, default=0, help='minibatch size (0 = full batch; their code: 1); --iters = steps')
ap.add_argument('--protocol', choices=['cf', 'firstint'], default='cf',
                help="cf: encode the factual regimen, predict others (Baaz's test). firstint: the iScience format -- the "
                     "encoder sees only the first dosing interval of a regimen, the model predicts the rest of the SAME regimen; "
                     "test patients observed under every regimen (own noise stream seed + 4242, identical across models); "
                     "scored on [second dose, 30 h]")
ap.add_argument('--no-check', action='store_true', help='skip the comparison with the EB-VAE run CSVs')
ap.add_argument('--euler-dt', type=float, default=0.25,   # 19/10 opt-in: NMI Euler step (h); 1/12 puts the 20/40-min visit samples on the grid
                help='forward-Euler step of the NMI model in hours (default 0.25, unchanged)')
ap.add_argument('--nmi-x0', choices=['encoded', 'zero'], default='encoded',   # 22/10 opt-in (review, job 3)
                help="NMI: initial PK state. encoded (default, unchanged): the encoder emits the initial concentration (Baaz's patients "
                     "start with drug on board). zero: their InitState = 0, faithful when patients start drug-free (--sim tacro)")
ap.add_argument('--sim', choices=['baaz', 'tacro'], default='baaz',
                help='tacro = the tacrolimus transposition (scripts/regimen/gen_tacro_regimens.py, as dh_regimen.py --sim tacro: '
                     'same patients for a given seed); CYP3A5 and hematocrit / 35 are appended to the encoder inputs as two '
                     'constant channels. Opt-in 2026-10-13; baaz = unchanged')
cli = ap.parse_args()
if cli.sim == 'tacro':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gen_tacro_regimens as G
    cli.no_check = True
NA, DESIGN, seed = cli.n_per_arm, cli.design, cli.seed
torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
t_start = time.time()
SAMP = np.arange(0, 30.25, 0.5) if DESIGN == 'dense' else np.array([0, 2, 7.5, 15.5, 23.5, 28.0])
EG = np.arange(0, 30.0001, 0.25)
TRAIN_TIMES = [0, 8, 16, 24]
TMAX, DMAX = 30.0, 800.0
LO, HI = 400.0, 800.0
if cli.sim == 'tacro':
    SAMP = G.SAMP_BY_DESIGN[DESIGN]
    EG, TRAIN_TIMES, (LO, HI) = G.EG, list(G.TRAIN_TIMES), G.ARMS
    TMAX, DMAX = G.T_END, HI

# ---------------------------------------------------------------- data: run_regimen.py's exact sequence (as dh_regimen*.py)
def sample_obs(P, times, amounts, rng):
    t_eval, c = G.simulate(P, times, amounts)
    return t_eval, c, G.noisy(np.stack([np.interp(SAMP, t_eval, ci) for ci in c]), rng)

rng = np.random.default_rng(seed)
n_tr = 2 * NA; P_tr = G.draw(n_tr, rng)
if cli.titrate is None:
    arm_tr, titr_corr = np.where(np.arange(n_tr) % 2 == 0, LO, HI), None
else:
    arm_tr, titr_corr = G.titrate(P_tr, cli.titrate, seed)
    print(f'TITRATED training arms: rho {cli.titrate}, corr(log ke, 800 mg) {titr_corr:+.2f}, '
          f'{int((arm_tr == HI).sum())}/{n_tr} on the high dose', flush=True)
A_tr = np.repeat(arm_tr[:, None], 4, 1); _, _, y_tr = sample_obs(P_tr, TRAIN_TIMES, A_tr, rng)
n_te = 400; P_te = G.draw(n_te, rng); arm_te = np.where(np.arange(n_te) % 2 == 0, LO, HI)
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
N_FIRST = int((SAMP < t2_of(TRAIN_TIMES)).sum())   # max samples in a first interval (decoder context length, padded)
YMAX = float(y_tr.max())
print(f'{n_tr} train patients, {len(SAMP)} samples each, YMAX {YMAX:.1f}, test {n_te}', flush=True)


def tens(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


def covs(P):
    """--sim tacro: [CYP3A5 expresser, hematocrit / 35] per patient; nothing on Baaz's simulator"""
    return np.stack([P[:, 6], P[:, 7] / 35.0], 1) if cli.sim == 'tacro' else np.zeros((len(P), 0))
NCOV = covs(P_tr[:1]).shape[1]


def rows(y, times, amounts, cut=None, cov=None):
    """Data rows on the grid SAMP U dose times (shared by all patients): [time since last dose, time, doses so far,
    amount dosed at that point, observed PK (0 where not sampled)], normalised. Returns [B, n_grid, 5] and the grid."""
    tt = np.asarray(times, float); grid = np.unique(np.concatenate([SAMP, tt])); B = len(y)
    last = np.array([tt[tt <= g].max() if (tt <= g).any() else 0.0 for g in grid])
    cyc = np.array([(tt <= g).sum() for g in grid], float)
    amt = np.zeros((B, len(grid)))
    for k, t in enumerate(tt):
        amt[:, np.searchsorted(grid, t)] += amounts[:, k] / DMAX
    pk = np.zeros((B, len(grid))); pk[:, np.searchsorted(grid, SAMP)] = y / YMAX
    if cut is not None:                      # PK only in the first dosing interval ('PK_cycle1 padded with zero')
        pk[:, grid >= cut] = 0.0
    bc = lambda v: np.broadcast_to(v, (B, len(grid)))
    out = np.stack([bc((grid - last) / TMAX), bc(grid / TMAX), bc(cyc / 4), amt, pk], -1)
    if cov is not None and cov.shape[1]:          # constant covariate channels (opt-in --sim tacro)
        out = np.concatenate([out, np.broadcast_to(cov[:, None, :], (B, len(grid), cov.shape[1]))], -1)
    return tens(out), grid


# ---------------------------------------------------------------- iScience model
class ISModel(nn.Module):
    def __init__(self, n_ctx, latent=6, gru=128, ode_hidden=16, dec_hidden=32):
        super().__init__()
        self.latent = latent
        self.gru = nn.GRU(5 + NCOV, gru, batch_first=True)
        self.enc_out = nn.Linear(gru, 2 * latent)
        layers, d = [], latent
        for _ in range(3):
            layers += [nn.Linear(d, ode_hidden), nn.SELU()]; d = ode_hidden
        self.ode = nn.Sequential(*layers, nn.Linear(d, latent))
        for m in self.ode:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.001); nn.init.constant_(m.bias, 0.5)
        self.dec = nn.Sequential(nn.Linear(latent + n_ctx, dec_hidden), nn.SELU(), nn.Linear(dec_hidden, 1))

    def encode(self, feats):
        _, h = self.gru(torch.flip(feats, [1]))            # scans the series from the end to the start
        out = self.enc_out(h[-1])
        return out[:, :self.latent], out[:, self.latent:].abs()

    def field(self, t, z):
        return self.ode(z)

    def rollout(self, z0, dose_times, amounts, t_out):
        """z0 [B,L]; each dose adds amount / 800 mg to latent coordinate 0 AFTER the output at its time (their loop: the
        prediction at t_i comes from z_{t_{i-1}} + dose at t_{i-1}, integrated to t_i). Returns z at t_out [B,T,L]."""
        a = tens(amounts / DMAX); t_out = [float(t) for t in t_out]; outs = {}

        def integrate(z, ta, tb, seg):
            grid = sorted(set([ta] + seg + [tb]))
            if len(grid) == 1:
                for t in seg:
                    outs[t] = z
                return z
            sol = odeint(self.field, z, torch.tensor(grid, dtype=torch.float32) / TMAX, rtol=1e-3, atol=1e-4)
            for t in seg:
                outs[t] = sol[grid.index(t)]
            return sol[-1]

        z, tc = z0, 0.0
        for k, tk in enumerate(float(t) for t in dose_times):
            if tk > t_out[-1]:
                break
            z = integrate(z, tc, tk, [t for t in t_out if tc <= t <= tk and t not in outs])
            z = z + F.pad(a[:, k:k + 1], (0, self.latent - 1))
            tc = tk
        integrate(z, tc, t_out[-1], [t for t in t_out if t >= tc and t not in outs])
        return torch.stack([outs[t] for t in t_out], 1)

    def decode(self, zt, ctx):
        return self.dec(torch.cat([zt, ctx.unsqueeze(1).expand(-1, zt.shape[1], -1)], -1)).squeeze(-1)


# ---------------------------------------------------------------- NMI model
DT = cli.euler_dt   # 0.25 unless --euler-dt


class NMIModel(nn.Module):
    def __init__(self, n_params=4, gru=20, hidden=32):
        super().__init__()
        self.gru = nn.GRU(4 + NCOV, gru, batch_first=True)
        self.head = nn.Linear(gru, n_params + 1)
        self.n_params = n_params
        self.pkvf = nn.Sequential(nn.Linear(2 + n_params, hidden), nn.Softplus(), nn.Linear(hidden, hidden), nn.Softplus(),
                                  nn.Linear(hidden, 2))
        nn.init.zeros_(self.pkvf[-1].weight); nn.init.zeros_(self.pkvf[-1].bias)   # start still (as our B): no blow-up / dead ReLU

    def encode(self, feats):
        """feats: the iScience rows minus the cycle channel -> [time after dose, time, PK, amount dosed]"""
        _, h = self.gru(feats[..., [0, 1, 4, 3] + list(range(5, 5 + NCOV))])
        out = self.head(h[-1])
        # the dosed (gut) coordinate starts EMPTY as their InitState = 0; only the initial concentration is encoded
        if cli.nmi_x0 == 'zero':
            return out[:, :self.n_params], torch.zeros(out.shape[0], 2)
        return out[:, :self.n_params], F.pad(F.softplus(out[:, self.n_params:]), (1, 0))

    def simulate(self, theta, x0, dose_times, amounts, t_out):
        """forward Euler on the 0.25 h grid; dose increments the dosed coordinate; ReLU; concentration = coordinate 1"""
        n = int(round(float(t_out[-1]) / DT)); a = tens(amounts / DMAX)
        dose_at = {}
        for k, t in enumerate(dose_times):
            if t <= t_out[-1]:
                dose_at.setdefault(int(round(float(t) / DT)), []).append(k)
        out_at = {int(round(float(t) / DT)): j for j, t in enumerate(t_out)}
        assert all(abs(s * DT - float(t_out[j])) < 1e-9 for s, j in out_at.items()), 'output times must be on the Euler grid'
        x, outs = x0, [None] * len(t_out)
        for s in range(n + 1):
            for k in dose_at.get(s, []):
                x = x + F.pad(a[:, k:k + 1], (0, 1))
            if s in out_at:
                outs[out_at[s]] = x[:, 1]
            if s < n:
                x = F.relu(x + DT * self.pkvf(torch.cat([x, theta], -1)))        # step in hours (their 1/4 day <-> 1/4 h)
        return torch.stack(outs, 1)


# ---------------------------------------------------------------- training
def ctx_first(y, cut):
    """decoder context: the first interval's (time, PK), zero-padded to N_FIRST samples"""
    m = SAMP < cut; t = np.zeros((len(y), N_FIRST)); v = np.zeros((len(y), N_FIRST))
    t[:, :m.sum()] = SAMP[m] / TMAX; v[:, :m.sum()] = y[:, m] / YMAX
    return tens(np.concatenate([t, v], 1))


def nmi_rows(f, grid, cut):
    return f[:, torch.as_tensor(grid < cut)] if cut is not None else f


CUT_TR = t2_of(TRAIN_TIMES) if FI else None
feats_tr, grid_tr = rows(y_tr, TRAIN_TIMES, A_tr, CUT_TR, covs(P_tr))
feats_te, _ = rows(y_te, TRAIN_TIMES, A_te, cov=covs(P_te))
if FI:
    ctx_tr = ctx_first(y_tr, CUT_TR); ctx_te = None
else:
    ctx_tr = tens(np.concatenate([np.broadcast_to(SAMP / TMAX, y_tr.shape), y_tr / YMAX], 1))
    ctx_te = tens(np.concatenate([np.broadcast_to(SAMP / TMAX, y_te.shape), y_te / YMAX], 1))
Y_tr = tens(y_tr / YMAX)
model = ISModel(ctx_tr.shape[1]) if cli.model == 'is' else NMIModel(hidden=cli.hidden)


def predict_train(idx=None):
    """training prediction at the samples (all of them; the encoder sees the first interval only under firstint)"""
    f, c, A = (feats_tr, ctx_tr, A_tr) if idx is None else (feats_tr[idx], ctx_tr[idx], A_tr[idx.numpy()])
    if cli.model == 'is':
        mu, sd = model.encode(f)
        z0 = mu + sd * torch.randn_like(mu)
        return model.decode(model.rollout(z0, TRAIN_TIMES, A, SAMP), c)
    theta, x0 = model.encode(nmi_rows(f, grid_tr, CUT_TR))
    return model.simulate(theta, x0, TRAIN_TIMES, A, SAMP)


def main():
    opt = torch.optim.Adam(model.parameters(), lr=cli.lr, weight_decay=cli.wd)
    gen = torch.Generator().manual_seed(seed)
    name = 'iScience (Lu, Deng, Zhang, Liu & Guan 2021)' if cli.model == 'is' else 'NMI (Lu, Bender, Jin & Guan 2021)'
    print(f'model: {name}, {sum(p.numel() for p in model.parameters())} params, {cli.iters} iterations', flush=True)
    for it in range(1, cli.iters + 1):
        for g in opt.param_groups:
            g['lr'] = max(cli.lr * 0.9995 ** (it * cli.batch / n_tr if cli.batch else it), cli.lr / 10)   # per epoch
        opt.zero_grad()
        idx = torch.randint(0, n_tr, (cli.batch,), generator=gen) if cli.batch else None
        loss = ((predict_train(idx) - (Y_tr if idx is None else Y_tr[idx])) ** 2).mean()
        if not torch.isfinite(loss):
            print(f'iteration {it}: non-finite loss, stopping', flush=True); break
        loss.backward(); opt.step()
        if it % (250 if not cli.batch else 5000) == 0 or it == 1:
            print(f'it {it:5d}  MSE {loss.item():9.5f}  train RMSE {loss.item() ** 0.5 * YMAX:7.2f} mg/L  '
                  f'({(time.time() - t_start) / 60:.1f} min)', flush=True)
    minutes_train = (time.time() - t_start) / 60

    res = dict(**({'sim': 'tacro', 'tacro_scenario': G.SCENARIO, 'arms_mg': [LO, HI]} if cli.sim == 'tacro' else {}), model='lu-iscience' if cli.model == 'is' else 'lu-nmi', N_per_arm=NA, design=DESIGN, seed=seed,
               n_samples_per_patient=len(SAMP), iters=cli.iters, lr=cli.lr, hidden=cli.hidden, wd=cli.wd, batch=cli.batch,
               protocol=cli.protocol, titrate=cli.titrate, titrate_corr=titr_corr, **({'nmi_x0': cli.nmi_x0} if cli.nmi_x0 != 'encoded' else {}),
               minutes_train=minutes_train, individual={})
    if cli.model == 'is':
        res['individual_factual_encoder'] = {}; res['identity'] = {}
    model.eval()

    def score(pred, name, cut=0.0):
        """AUC and curve error on [cut, 30 h] (cut = 0: the whole 0-30 h window)"""
        w = EG >= cut - 1e-9
        tru = np.stack([np.interp(EG, t_eval, c) for c in truth[name]])[:, w]; pred = pred[:, w]
        auc_p, auc_t = np.trapezoid(pred, EG[w], axis=1), np.trapezoid(tru, EG[w], axis=1)
        r = auc_p / auc_t - 1; nrmse = np.sqrt(np.mean((pred - tru) ** 2, 1)) / np.mean(tru, 1)
        return dict(auc_abs_median=100 * float(np.median(np.abs(r))), nrmse_median=100 * float(np.median(nrmse)),
                    auc_bias_median=100 * float(np.median(r)), auc_err_per_patient=[float(x) for x in r],
                    auc_pred_per_patient=[float(x) for x in auc_p])

    if FI:
        res['window_start'] = {}
        with torch.no_grad():
            for name in G.REGIMENS:
                times, A = regs[name]; cut = t2_of(times)
                f_r, grid = rows(y_obs[name], times, A, cut, covs(P_te))
                if cli.model == 'is':
                    mu, _ = model.encode(f_r)
                    pred = model.decode(model.rollout(mu, times, A, EG), ctx_first(y_obs[name], cut)).numpy() * YMAX
                else:
                    theta, x0 = model.encode(nmi_rows(f_r, grid, cut))
                    pred = model.simulate(theta, x0, times, A, EG).numpy() * YMAX
                res['individual'][name] = s = score(pred, name, cut); res['window_start'][name] = cut
                print(f'{name:8s} first {cut:g} h observed -> |AUC({cut:g}-30 h) err| {s["auc_abs_median"]:5.1f}%  '
                      f'nRMSE {s["nrmse_median"]:5.1f}%  bias {s["auc_bias_median"]:+5.1f}%', flush=True)
        res.pop('individual_factual_encoder', None); res.pop('identity', None)
    else:
     with torch.no_grad():
        if cli.model == 'is':
            mu_fact, _ = model.encode(feats_te)
            z0s = {}
            for name in G.REGIMENS:
                times, A = regs[name]
                f_r, _ = rows(y_te, times, A, cov=covs(P_te))     # SAME observations, the regimen's dosing channels
                z0s[name], _ = model.encode(f_r)
                for key, z0 in (('individual', z0s[name]), ('individual_factual_encoder', mu_fact)):
                    pred = model.decode(model.rollout(z0, times, A, EG), ctx_te).numpy() * YMAX
                    res[key][name] = score(pred, name)
                i, j = res['individual'][name], res['individual_factual_encoder'][name]
                print(f'{name:8s} |AUC err| target-encoder {i["auc_abs_median"]:5.1f}% (bias {i["auc_bias_median"]:+5.1f}%)   '
                      f'factual-encoder {j["auc_abs_median"]:5.1f}% (bias {j["auc_bias_median"]:+5.1f}%)', flush=True)
            # test 1: does the regimen being predicted move the patient's latent?
            from sklearn.linear_model import RidgeCV
            from sklearn.model_selection import cross_val_predict
            from sklearn.metrics import r2_score
            Z0 = z0s['train'].numpy(); den = 2 * np.trace(np.cov(Z0.T))
            PNAMES = ('log ka', 'log ke', 'log v')
            phys = np.log(P_te)
            if cli.sim == 'tacro':     # probe the PK parameters, not the covariates (CYP3A5 is 0/1); G.PROBE per scenario
                PNAMES, phys = tuple(G.PROBE), np.log(P_te[:, list(G.PROBE.values())])
            probe = {}
            for k, pname in enumerate(PNAMES):
                m = RidgeCV(alphas=np.logspace(-3, 3, 13))
                probe[pname] = dict(cv_r2_train=float(r2_score(phys[:, k], cross_val_predict(m, Z0, phys[:, k], cv=5))))
                m.fit(Z0, phys[:, k])
                for name in G.REGIMENS:
                    probe[pname][name] = float(r2_score(phys[:, k], m.predict(z0s[name].numpy())))
            for name in G.REGIMENS:
                res['identity'][name] = float(np.mean(np.sum((z0s[name].numpy() - Z0) ** 2, 1)) / den)
            res['identity_probe'] = probe
            print('identity ratio vs train-regimen z0: ' + ', '.join(f'{k} {v:.3f}' for k, v in res['identity'].items()), flush=True)
            for pname, d in probe.items():
                print(f'  probe {pname}: CV R^2 {d["cv_r2_train"]:.3f}; applied to z0(regimen): ' +
                      ', '.join(f'{k} {d[k]:.3f}' for k in G.REGIMENS), flush=True)
        else:
            theta, x0 = model.encode(feats_te)
            for name in G.REGIMENS:
                times, A = regs[name]
                pred = model.simulate(theta, x0, times, A, EG).numpy() * YMAX
                res['individual'][name] = s = score(pred, name)
                print(f'{name:8s} |AUC err| {s["auc_abs_median"]:5.1f}%  nRMSE {s["nrmse_median"]:5.1f}%  bias {s["auc_bias_median"]:+5.1f}%', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    json.dump(res, open(cli.out, 'w'), indent=1)
    torch.save(dict(state_dict=model.state_dict(), cli=vars(cli), YMAX=YMAX), os.path.splitext(cli.out)[0] + '.pt')
    print(f'REGIMEN_LU_DONE {(time.time() - t_start) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
