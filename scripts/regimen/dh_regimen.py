#!/usr/bin/env python3
"""Baaz et al. 2026's own regimen test with OUR model. Same design, same patients and same metrics as the EB-VAE runs of
third_party/baaz_ebvae/ours/regimen/run_regimen.py:
  their LINEAR 1-cpt simulator (gen_regimens.py: correlated log-normal ka, ke, v; central amount starts at v; doses into the
  gut; additive noise sd 10), training arms 400 / 800 mg x 4 doses at 0, 8, 16, 24 h, observed 0-30 h either 'dense'
  (every 0.5 h, 61) or 'sparse' (0, 2, 7.5, 15.5, 23.5, 28 h); 400 NEW test patients encoded from their own training-
  regimen data, predicted under train | amt350 / amt600 / amt850 | time400 / time800 (3 doses at 0, 3, 8 h);
  individual: posterior-mean latent, median |AUC(0-30) err| / nRMSE / bias vs the same patient's noise-free curve;
  population: 2000 prior draws vs 2000 new true patients, mean |error| of the 10/50/90 % quantile curves / true median mean.
  The data are regenerated with run_regimen.py's exact random-number sequence and checked against the CSVs that run wrote.

Model = variant B generalised to arbitrary dosing (the design that can dose at any time): main FiLM-config latent ODE
(ODE-RNN encoder, -l 10, 100-unit flow, residual 50-unit decoder) with --orth-split 1,5 (drug subspace 6 dims, physiology
4); each dose = translation z -> z + (amount / 800 mg) * b inside the drug subspace; the flow only moves the drug
coordinates, vanishes at drug coordinate 0 for every physiology, carries a linear elimination term, and its output layer
starts at zero (lib/create_latent_ode_model.py, --dose-history). Differences from the tacrolimus B, forced by this design:
  * the INITIAL drug state is encoded (z_start = z0, not P_phys z0): their patients start with drug on board (A_c(0) = v);
  * doses are an arbitrary list of (time, amount), solved event by event (a dose at t applies before the output at t,
    as in their simulator);
  * observations are linear-scale (/ max training observation, their noise is additive) with a LEARNED noise sd;
  * encoder input = [y, cumulative dose so far] at the observed times, static = [arm dose / 800 mg].
  --plain: the same latent ODE WITHOUT the B structure (dose kick b in the full latent space, unconstrained flow, default
  init) -- does the structure matter for unseen regimens?
Training: full batch, Adamax lr 0.01 decayed 0.999 / iteration to 0.001, KL weight 0 for 20 iterations then 1 - 0.99^k
(the main config's schedules), IWAE over 3 posterior samples.

    dh_regimen.py <n_per_arm> <dense|sparse> <seed> <out.json> [--iters 3000] [--plain] [--latents 10] [--split 1,5]"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, time, argparse, copy
import numpy as np, torch
sys.path.insert(0, os.getcwd())
sys.path.insert(0, f'{REPO}/third_party/baaz_ebvae/ours/regimen')
import gen_regimens as G
from probe_entanglement import _priors
from lib.create_latent_ode_model import create_LatentODE_model

ap = argparse.ArgumentParser()
ap.add_argument('n_per_arm', type=int); ap.add_argument('design', choices=['dense', 'sparse', 'visit', 'visit1'])   # visit / visit1: --sim tacro only
ap.add_argument('seed', type=int); ap.add_argument('out')
ap.add_argument('--iters', type=int, default=3000)
ap.add_argument('--plain', action='store_true')
ap.add_argument('--latents', type=int, default=10)
ap.add_argument('--split', default='1,5')
ap.add_argument('--no-check', action='store_true', help='skip the comparison with the EB-VAE run CSVs')
ap.add_argument('--obsrv-std', type=float, default=None,
                help='FIXED observation noise sd (units of the max training observation; B2 uses 0.05). Unset = learned '
                     '(the original B). Opt-in 2026-10-08: free-flow B with fixed noise')
ap.add_argument('--obsrv-mgl', type=float, default=None,
                help='FIXED observation noise sd in mg/L (converted with this run\'s YMAX; overrides --obsrv-std). Opt-in 2026-10-09')
ap.add_argument('--kick-phys', type=int, default=0,
                help='hidden width H > 0: physiology-dependent LINEAR jump z -> z + (a/800) Q_drug [b + g(p)], g = MLP(4 -> H -> 6) '
                     'of the physiology coordinates, last layer zero-initialised (starts at the population kick b). '
                     '0 = off (the original B). Opt-in 2026-10-08')
ap.add_argument('--protocol', choices=['cf', 'firstint'], default='cf',
                help='firstint = the iScience format: the encoder sees only the first dosing interval, the model predicts '
                     'the rest of the SAME regimen; test patients observed under every regimen (stream seed + 4242); '
                     'scored on [second dose, 30 h]')
ap.add_argument('--titrate', type=float, default=None,
                help='titrated training arms (gen_regimens.titrate: fast eliminators get 800 mg; own random stream, so '
                     'patients and noise are those of the randomised run); test arms unchanged')
ap.add_argument('--enc-dose', choices=['raw', 'norm', 'none', 'normd'], default='raw',
                help='what the ENCODER is told about the dose (dynamics always get the true doses), as dh_regimen_v3.py: raw = '
                     'concentrations + cumulative dose + arm dose (default); norm = concentrations divided by the dose (relative '
                     'to 800 mg), dose count only, no dose value; none = concentrations + dose count only; normd = '
                     'norm + the raw encoder\'s dose inputs (cumulative dose channel, arm dose in static) through weights that '
                     'start at zero, so training starts from norm (opt-in 2026-10-15)')
ap.add_argument('--dose-cond', action='store_true',
                help='DOSE-CONDITIONED baseline (opt-in 2026-10-12): no dose kick, no orth split; the vector field '
                     'receives the dose as an input, dz/dt = f(z, u(t)), u(t) = sum_k (a_k / 800 mg) exp(-(t - t_k) / tau) over the '
                     'doses given so far, tau learned (init 1 h). Same encoder, latent size, decoder, likelihood and training')
ap.add_argument('--dc-zero-init', action='store_true',
                help='dose-cond only (tuning): zero-initialise the output layer of the dose-conditioned field, so the '
                     'field starts at 0 and the latent does not leave its encoded start at initialisation (the default std-0.1 init '
                     'gives O(0.5)/h fields over a 30-45 h window: initial prediction errors of 700 mg/L, then a plateau)')
ap.add_argument('--readout-relu', action='store_true',   # 19/10 opt-in: non-negative concentrations (12-sample dips below zero)
                help='clip the readout at zero (concentrations cannot be negative)')
ap.add_argument('--readout-softplus', type=float, default=0.0,   # 22/10 opt-in (review): positive readout without the dead gradients of the ReLU
                help='BETA > 0: concentration = softplus(BETA * h) / BETA on the normalised scale (positive, ~h above a few 1/BETA)')
ap.add_argument('--depot', type=int, default=0,   # 19/10 opt-in: dose into K drug coordinates that the readout does not see
                help='K > 0: each dose translates only the first K drug coordinates (a depot), which the readout ignores, so the '
                     'concentration cannot jump at a dose; the learned flow carries the drug to the observed coordinates')
ap.add_argument('--depot-linear', action='store_true',   # 19/10 opt-in: --depot 1 learned two dose levels as a convex transfer
                help='with --depot 1: the depot empties at a first-order rate ka(p) set by the physiology into a learned direction of the '
                     'other drug coordinates, and the flow network neither reads nor moves it, so the absorbed amount is proportional '
                     'to the dose (the elimination stays learned)')
ap.add_argument('--sim', choices=['baaz', 'tacro'], default='baaz',
                help='tacro = the tacrolimus transposition of this benchmark (scripts/regimen/gen_tacro_regimens.py: the paper\'s '
                     'scenario-4 simulator, Prograf q12h, arms 2 / 4 mg, CYP3A5 and hematocrit given to the encoder as static '
                     'inputs); opt-in 2026-10-13, baaz = their linear simulator (default, unchanged)')
cli = ap.parse_args()
if cli.design in ('visit', 'visit1') and cli.sim != 'tacro':
    sys.exit("designs 'visit' / 'visit1' (intensive PK visit) exist only for --sim tacro")
if cli.readout_relu and cli.readout_softplus:
    sys.exit('--readout-relu and --readout-softplus exclude each other')
if cli.sim == 'tacro':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gen_tacro_regimens as G
    cli.no_check = True
if cli.dc_zero_init and not cli.dose_cond:
    sys.exit('--dc-zero-init applies to --dose-cond only')
if cli.dose_cond and (cli.plain or cli.kick_phys > 0):
    sys.exit('--dose-cond excludes --plain and --kick-phys')
NA, DESIGN, seed = cli.n_per_arm, cli.design, cli.seed
torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
dev = torch.device('cpu'); t_start = time.time()
SAMP = np.arange(0, 30.25, 0.5) if DESIGN == 'dense' else np.array([0, 2, 7.5, 15.5, 23.5, 28.0])
EG = np.arange(0, 30.0001, 0.25)
TRAIN_TIMES = [0, 8, 16, 24]
LO, HI = 400.0, 800.0                       # training arms (mg)
if cli.sim == 'tacro':
    SAMP = G.SAMP_BY_DESIGN[DESIGN]
    EG, TRAIN_TIMES, (LO, HI) = G.EG, list(G.TRAIN_TIMES), G.ARMS

# ---------------------------------------------------------------- data: run_regimen.py's exact sequence
def sample_obs(P, times, amounts, rng):
    t_eval, c = G.simulate(P, times, amounts)
    return t_eval, c, G.noisy(np.stack([np.interp(SAMP, t_eval, ci) for ci in c]), rng)

rng = np.random.default_rng(seed)
n_tr = 2 * NA; P_tr = G.draw(n_tr, rng)
if cli.titrate is None:
    arm_tr, titr_corr = np.where(np.arange(n_tr) % 2 == 0, LO, HI), None
else:
    arm_tr, titr_corr = G.titrate(P_tr, cli.titrate, seed)
    print(f'TITRATED training arms: rho {cli.titrate}, corr(log ke, 800 mg) {titr_corr:+.2f}', flush=True)
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

YMAX = float(y_tr.max()); DMAX = HI
if cli.obsrv_mgl is not None:   # absolute noise level, same in mg/L for every seed / design
    cli.obsrv_std = cli.obsrv_mgl / YMAX
print(f'{n_tr} train patients, {len(SAMP)} samples each, YMAX {YMAX:.1f}, test {n_te}', flush=True)

def tens(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)

def covs(P):
    """static covariates appended for --sim tacro: [CYP3A5 expresser, hematocrit / 35]; none for Baaz's simulator"""
    return np.stack([P[:, 6], P[:, 7] / 35.0], 1) if cli.sim == 'tacro' else np.zeros((len(P), 0))

def enc_inputs(y, amounts, times, cov=None):
    """[y / YMAX, cumulative dose so far / (4 * DMAX)] at the sample times; static = [arm dose / DMAX] (+ cov)"""
    cum = np.stack([(amounts * (np.array(times)[None, :] <= t)).sum(1) for t in SAMP], 1) / (4 * DMAX)
    cov = np.zeros((len(y), 0)) if cov is None else cov
    if cli.enc_dose == 'raw':
        data = np.stack([y / YMAX, cum], -1)
        return tens(data), tens(np.concatenate([amounts[:, :1] / DMAX, cov], 1))
    count = np.stack([(np.array(times)[None, :] <= t).sum(1).repeat(len(y)) for t in SAMP], 1) / 4.0   # doses so far, no amount
    scale = amounts[:, :1] / DMAX if cli.enc_dose in ('norm', 'normd') else np.ones((len(y), 1))
    if cli.enc_dose == 'normd':   # extra LAST data channel and LAST static input, read through zero-initialised weights
        return (tens(np.stack([y / YMAX / scale, count, cum], -1)),
                tens(np.concatenate([np.ones((len(y), 1)), cov, amounts[:, :1] / DMAX], 1)))
    return tens(np.stack([y / YMAX / scale, count], -1)), tens(np.concatenate([np.ones((len(y), 1)), cov], 1))

X_tr, S_tr = enc_inputs(y_tr, A_tr, TRAIN_TIMES, covs(P_tr))
X_te, S_te = enc_inputs(y_te, A_te, TRAIN_TIMES, covs(P_te))
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
            X_r, S_r = enc_inputs(y_obs[name], A, times, covs(P_te))
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

# ---------------------------------------------------------------- model
T = 'confound_vc00_s4_pnoise_win_n100'
import argparse as _argparse, json as _json
args = _argparse.Namespace(**_json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dose_event_base_args.json'))))   # architecture flags of the model
args.use_film = False; args.latents = cli.latents; args.static_dim = 1 + covs(P_tr[:1]).shape[1]
args.orth_split = None if (cli.plain or cli.dose_cond) else cli.split
args.dose_history = 0 if (cli.plain or cli.dose_cond) else 1; args.dose_interval = 24.0; args.dose_interval_prograf = 0.0
obsrv_std, z0_prior = _priors(dev, 0.01)
model = create_LatentODE_model(args, 1, z0_prior, obsrv_std, dev)
ode = model.diffeq_solver.ode_func
if cli.dose_cond:   # field on [z, u(t)]: same layers as the generative flow, one extra input; own RNG fork
    from lib import utils as _U
    with torch.random.fork_rng():
        torch.manual_seed(seed + 53)
        model.dc_net = _U.create_net(cli.latents + 1, cli.latents, n_layers=args.gen_layers, n_units=args.units,
                                     nonlinear=torch.nn.Tanh)
        _U.init_network_weights(model.dc_net)
        if cli.dc_zero_init:   # no RNG consumed: default path unchanged
            torch.nn.init.zeros_(model.dc_net[-1].weight); torch.nn.init.zeros_(model.dc_net[-1].bias)
    model.dc_log_tau = torch.nn.Parameter(torch.tensor(0.0))
    _DC = dict(applied=[])   # [(t_k, a_k [B])] doses given before the current segment

    def _dc_grad(t_local, y):
        u = torch.zeros(y.shape[:-1] + (1,))
        tau = model.dc_log_tau.exp()
        for t_k, a_k in _DC['applied']:
            u = u + (a_k * torch.exp(-(t_local - t_k) / tau)).reshape(1, -1, 1)
        return model.dc_net(torch.cat([y, u], -1))
    ode.get_ode_gradient_nn = _dc_grad
if cli.plain:   # dose kick in the full latent space, unconstrained flow
    model.kick_full = torch.nn.Parameter(torch.full((cli.latents,), 1.0 / cli.latents ** 0.5))
if cli.kick_phys > 0:   # b(p): own RNG fork, so the global stream (z0 samples, ...) is the same as without it
    if cli.plain:
        sys.exit('--kick-phys needs the B structure (not --plain)')
    _kfd = model.orth_split.k_f + model.orth_split.k_d
    with torch.random.fork_rng():
        torch.manual_seed(seed + 31)
        model.kick_net = torch.nn.Sequential(torch.nn.Linear(cli.latents - _kfd, cli.kick_phys), torch.nn.Tanh(),
                                             torch.nn.Linear(cli.kick_phys, _kfd))
    with torch.no_grad():
        model.kick_net[-1].weight.zero_(); model.kick_net[-1].bias.zero_()
_DL = {}
if cli.depot_linear:   # linear first-order depot; ka_net on its own RNG fork, so the global stream is the same as without it
    if cli.depot != 1 or cli.kick_phys <= 0:
        sys.exit('--depot-linear needs --depot 1 and --kick-phys')
    with torch.random.fork_rng():
        torch.manual_seed(seed + 71)
        model.ka_net = torch.nn.Sequential(torch.nn.Linear(cli.latents - _kfd, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
    with torch.no_grad():   # starts at ka = 0.5 /h for every patient
        model.ka_net[-1].weight.zero_(); model.ka_net[-1].bias.fill_(float(np.log(np.expm1(0.5))))
    model.depot_dir = torch.nn.Parameter(torch.full((_kfd - 1,), 1.0 / (_kfd - 1) ** 0.5))
    _flow = ode.get_ode_gradient_nn

    def _dl_grad(t_local, y):
        Q = model.orth_split.Q(); q = Q[:, :1]
        d = y @ q                                   # [S,B,1] depot content
        g = _flow(t_local, y - d @ q.t())           # anchored flow + elimination, blind to the depot
        g = g - (g @ q) @ q.t()                     # and not moving it
        return g + _DL['ka'] * d * (Q[:, 1:_kfd] @ model.depot_dir - q[:, 0])
    ode.get_ode_gradient_nn = _dl_grad
if cli.enc_dose == 'normd':   # widen the encoder's input layers by one zero column each; own RNG fork (no global draws)
    def _widen(lin):
        with torch.random.fork_rng():
            new = torch.nn.Linear(lin.in_features + 1, lin.out_features)
        with torch.no_grad():
            new.weight.zero_(); new.weight[:, :-1] = lin.weight; new.bias.copy_(lin.bias)
        return new
    _enc = model.encoder_z0
    for _seq in (_enc.GRU_update.update_gate, _enc.GRU_update.reset_gate, _enc.GRU_update.new_state_net, _enc.static_encoder):
        _seq[0] = _widen(_seq[0])
    _enc.input_dim += 1; _enc.static_dim += 1
log_sd = torch.nn.Parameter(torch.tensor(np.log(0.05 if cli.obsrv_std is None else cli.obsrv_std), dtype=torch.float32),
                            requires_grad=cli.obsrv_std is None)   # learned observation noise (normalised units); fixed with --obsrv-std


def kick_vec():
    return model.kick_full if cli.plain else model.dose_hist.kick()


def simulate(z0, dose_times, amounts, t_out):
    """z0 [S,B,D] = state at t=0 before any dose; dose_times shared (sorted), amounts [B,K] mg; t_out 1-D sorted (>= 0).
    Returns latent trajectory [S,B,len(t_out),D]. A dose at t applies before the output at t (their simulator)."""
    t_out = [float(t) for t in t_out]
    kick = None if cli.dose_cond else kick_vec()
    if cli.kick_phys > 0:   # per-patient kick [S,B,D]; physiology is conserved, so read it from z0
        Q = model.orth_split.Q(); kfd = _kfd
        kick = (model.dose_hist.kick_coef + model.kick_net(z0 @ Q[:, kfd:])) @ Q[:, :kfd].t()
    if cli.depot > 0 and kick is not None:   # keep only the depot component of the jump
        Qd = model.orth_split.Q()[:, :cli.depot]
        kick = (kick @ Qd) @ Qd.t()
    a = tens(amounts / DMAX)
    outs = {}

    def integrate(z, ta, tb, seg):
        """solve from ta to tb, storing the state at every t in seg (ta <= t <= tb); returns the state at tb"""
        grid = sorted(set([ta] + seg + [tb]))
        if len(grid) == 1:
            for t in seg:
                outs[t] = z
            return z
        sol = model.diffeq_solver(z, torch.tensor(grid, dtype=torch.float32))
        for t in seg:
            outs[t] = sol[:, :, grid.index(t)]
        return sol[:, :, -1]

    if cli.dose_cond:
        _DC['applied'] = []
    elif not cli.plain:   # gradient_net at the drug-free point of this physiology: constant along the trajectory
        ode._anchor = ode.gradient_net(z0 @ model.orth_split.P_phys())
    if cli.depot_linear:   # physiology is conserved, so ka(p) is read from z0 like the jump
        _DL['ka'] = torch.nn.functional.softplus(model.ka_net(z0 @ model.orth_split.Q()[:, _kfd:]))
    try:
        z, tc = z0, 0.0
        for k, t_k in enumerate(float(t) for t in dose_times):
            if t_k > t_out[-1]:
                break
            z = integrate(z, tc, t_k, [t for t in t_out if tc <= t < t_k])
            if cli.dose_cond:   # the dose enters the field from t_k on (segments never cross a dose time)
                _DC['applied'] = _DC['applied'] + [(t_k, a[:, k])]
            else:
                z = z + a[:, k].reshape(1, -1, 1) * kick        # the dose applies before the output at t_k
            tc = t_k
        integrate(z, tc, t_out[-1], [t for t in t_out if t >= tc])
        return torch.stack([outs[t] for t in t_out], 2)
    finally:
        ode._anchor = None
        if cli.dose_cond:
            _DC['applied'] = []


def decode(traj):
    if cli.depot > 0:   # the readout does not see the depot coordinates
        Qd = model.orth_split.Q()[:, :cli.depot]
        traj = traj - (traj @ Qd) @ Qd.t()
    out = model.decoder(traj).squeeze(-1)    # [S,B,T] normalised concentration
    if cli.readout_softplus > 0:
        return torch.nn.functional.softplus(out, beta=cli.readout_softplus)
    return torch.relu(out) if cli.readout_relu else out


def main():
    params = list(model.parameters()) + [log_sd]
    opt = torch.optim.Adamax(params, lr=0.01)
    S = 3
    print(f'model: {"dose-conditioned field f(z, u(t))" if cli.dose_cond else "plain latent ODE + dose kicks" if cli.plain else "B (orth-split " + cli.split + ", dose kicks, anchored flow)"}, '
          f'latents {cli.latents}, {sum(p.numel() for p in params)} params, {cli.iters} iterations', flush=True)
    for it in range(1, cli.iters + 1):
        for g in opt.param_groups:
            g['lr'] = max(g['lr'] * 0.999, 0.001)
        kl_coef = 0.0 if it < 20 else 1 - 0.99 ** (it - 20)
        opt.zero_grad()
        mu, std = model.encoder_z0(X_enc_tr, tp_enc_tr, static=S_tr, run_backwards=True)
        std = std.abs().clamp(min=1e-5)
        z0 = mu + std * torch.randn(S, *mu.shape[1:])
        pred = decode(simulate(z0, TRAIN_TIMES, A_tr, SAMP))
        sd = log_sd.exp().clamp(min=1e-3)
        loglik = torch.distributions.Normal(pred, sd).log_prob(Y_tr.unsqueeze(0)).sum(-1)           # [S,B]
        kl = torch.distributions.kl_divergence(torch.distributions.Normal(mu, std), torch.distributions.Normal(0., 1.)).sum(-1)  # [1,B]
        iwae = torch.logsumexp(loglik - kl_coef * kl, 0) - np.log(S)
        loss = -iwae.mean() / len(SAMP)
        if not torch.isfinite(loss):
            print(f'iteration {it}: non-finite loss, stopping', flush=True); break
        loss.backward(); opt.step()
        if it % 250 == 0 or it == 1:
            with torch.no_grad():
                mse = ((pred.mean(0) - Y_tr) ** 2).mean().sqrt().item() * YMAX
            print(f'it {it:5d}  loss {loss.item():9.4f}  kl {kl.mean().item():7.3f}  noise sd {sd.item() * YMAX:6.2f}  '
                  f'train RMSE {mse:7.2f}  ({(time.time() - t_start) / 60:.1f} min)', flush=True)
    minutes_train = (time.time() - t_start) / 60

    # ---------------------------------------------------------------- evaluation (run_regimen.py's metrics)
    res = dict(**({'sim': 'tacro', 'tacro_scenario': G.SCENARIO, 'arms_mg': [LO, HI]} if cli.sim == 'tacro' else {}), model='dose-cond-field' if cli.dose_cond else ('B-dose-events' if not cli.plain else 'plain-dose-events'), N_per_arm=NA, design=DESIGN, seed=seed,
               n_samples_per_patient=len(SAMP), iters=cli.iters, latents=cli.latents, split=None if (cli.plain or cli.dose_cond) else cli.split,
               **({'dose_cond_tau_h': float(model.dc_log_tau.exp())} if cli.dose_cond else {}), **({'dc_zero_init': True} if cli.dc_zero_init else {}), minutes_train=minutes_train, **({'kick_phys': cli.kick_phys} if cli.kick_phys > 0 else {}), titrate=cli.titrate, titrate_corr=titr_corr, noise_sd=float(log_sd.exp() * YMAX), individual={}, population={}, **({"noise_fixed": True} if cli.obsrv_std is not None else {}), **({'enc_dose': cli.enc_dose} if cli.enc_dose != 'raw' else {}), **({'readout_relu': True} if cli.readout_relu else {}), **({'readout_softplus': cli.readout_softplus} if cli.readout_softplus > 0 else {}), **({'depot': cli.depot} if cli.depot else {}), **({'depot_linear': True} if cli.depot_linear else {}))
    model.eval()
    if FI:
        fi_eval(res, model.encoder_z0, lambda mu, times, A: decode(simulate(mu, times, A, EG))[0].numpy() * YMAX)
    else:
     with torch.no_grad():
        mu_te, _ = model.encoder_z0(X_te, tp_samp, static=S_te, run_backwards=True)
        P_pop = G.draw(2000, np.random.default_rng(seed + 777))
        for name in G.REGIMENS:
            times, A = regs[name]
            pred = decode(simulate(mu_te, times, A, EG))[0].numpy() * YMAX
            tru = np.stack([np.interp(EG, t_eval, c) for c in truth[name]])
            r = np.trapezoid(pred, EG, axis=1) / np.trapezoid(tru, EG, axis=1) - 1
            nrmse = np.sqrt(np.mean((pred - tru) ** 2, 1)) / np.mean(tru, 1)
            res['individual'][name] = dict(auc_abs_median=100 * float(np.median(np.abs(r))), nrmse_median=100 * float(np.median(nrmse)),
                                           auc_bias_median=100 * float(np.median(r)),
                                           auc_err_per_patient=[float(x) for x in r])
            A_pop = (np.repeat(np.where(np.arange(2000) % 2 == 0, LO, HI)[:, None], len(times), 1) if name == 'train'
                     else np.tile(A[:1], (2000, 1)))
            _, c_pop = G.simulate(P_pop, times, A_pop)
            q_true = np.percentile(np.stack([np.interp(EG, t_eval, c) for c in c_pop]), [10, 50, 90], 0)
            zp = torch.randn(1, 2000, cli.latents)
            sims = decode(simulate(zp, times, A_pop, EG))[0].numpy() * YMAX
            q_pred = np.percentile(sims, [10, 50, 90], 0)
            res['population'][name] = {f'q{q}_mae_rel': 100 * float(np.mean(np.abs(q_pred[j] - q_true[j])) / np.mean(q_true[1]))
                                       for j, q in enumerate((10, 50, 90))}
            print(f'{name:8s} individual |AUC err| {res["individual"][name]["auc_abs_median"]:5.1f}%  nRMSE '
                  f'{res["individual"][name]["nrmse_median"]:5.1f}%  bias {res["individual"][name]["auc_bias_median"]:+5.1f}%  | population '
                  f'q10/q50/q90 ' + ' / '.join(f'{v:.1f}%' for v in res['population'][name].values()), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)
    json.dump(res, open(cli.out, 'w'), indent=1)
    torch.save(dict(state_dict=model.state_dict(), log_sd=log_sd.detach(), args=args, cli=vars(cli), YMAX=YMAX),
               os.path.splitext(cli.out)[0] + '.pt')
    print(f'REGIMEN_DH_DONE {(time.time() - t_start) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
