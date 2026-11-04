#!/usr/bin/env python3
"""Null test of the a-posteriori physiology tracker.

    track_null.py <film|a|b> <ckpt> <out.json> [--n N] [--q 0.5] [--dref 4] [--sigmas eb,ckpt] [--iters 40]
    track_null.py --selftest            # LM + Laplace evidence against a closed-form linear-Gaussian case

Why a null test: the s4 test patients have two visits with IDENTICAL PK parameters at two doses (V1 the observed dose,
V2 a new one; 0/1/3 h samples at both). A tracker that estimates each visit's patient state should find no physiological
movement between them. Whatever it finds is false movement; its slope on log(d2/d1) is dose-map error leaking into
"physiology".

State of a visit. FiLM and A: the latent z0 at that visit's dose (L dims). B: the physiology coordinates p = Q_phys' z0
(n_p dims); the visit is generated from p through its dosing history, so p is the whole patient state.
Emission. The frozen model's prediction at 0/1/3 h, Gaussian with sd sigma in the model's Box-Cox units. sigma = 'eb'
(maximum Laplace marginal likelihood of the V1 samples of the first 300 test patients; V2 never used) or 'ckpt' (the
checkpoint's --obsrv-std).
Population prior of a visit's state. The aggregate posterior of the TRAINING cohort's encodings (both visits): mean of
the encoder means, covariance = their covariance + the mean encoder covariance. FiLM / A: the encodings are first
transported to d_ref by the model's FiLM (affine under --film-no-z0-cond) and the prior at a visit's dose d is the
push-forward through T_{d_ref -> d}, so the prior does not mix doses. B: p is dose-free.
Readout. g = log AUC at a reference regimen (--dref mg, the patient's own formulation): FiLM / A transport the state to
d_ref with the FiLM; B simulates the dosing history at d_ref.

Methods, same patients:
  ENC  encoder posterior mean of each visit, independently (amortised; what the model does today); sd from the
       encoder's own posterior covariance.
  IND  per-visit MAP: each visit's 3 samples + the population prior, independently.
  J0   one state for both visits, s2 = T_{d1->d2}(s1) (FiLM transport; identity for B): no movement allowed. For FiLM / A
       its Delta g is the transport's path inconsistency (T_{d2->ref} o T_{d1->d2} != T_{d1->ref}); for B it is 0.
  Jq   two-visit MAP smoother: s1 ~ prior at d1, s2 | s1 ~ N(T_{d1->d2}(s1), q^2 P_d2), P_d2 the prior covariance at d2.
MAP by batched Levenberg-Marquardt (Gauss-Newton Hessian; exact Jacobians by autograd through the dopri5 solve --
torchdiffeq computes the adaptive step under no_grad, so patients do not couple in the Jacobian; they share the step
sequence in the forward pass, as in every batched evaluation). Laplace: Cov = H^-1; sd of Delta g by the delta method;
log Bayes factor Jq vs J0 from the two Laplace evidences.

--phys-only (kind a only; opt-in, 2026-10-08): track A's PHYSIOLOGY BLOCK only. Everything outside it (FiLM coordinate,
dynamics start) is one shared state; between visits only a change w of the physiology coordinates p = Q_phys' z0 is
allowed: s2 = T_{d1->d2}(s1 + Q_phys w) (w = 0 is J0 exactly; the transport never touches p, so p2 = p1 + w). Runs ENC,
J0 and, per sigma:
  INDp  the visit-2 physiology gets its own population prior: p1 + w ~ N(m_p, C_p) (m_p, C_p = physiology marginal of the
        prior; dose-free under A).
  JPq   smoother on the physiology change: w ~ N(0, q^2 C_p).
Each is reported twice: '<m>_<sig>' with Delta = g_ref(s2, d2) - g_ref(s1, d1) (same definition as the other rows, so it
includes the FiLM path inconsistency that J0 measures) and '<m>c_<sig>' with Delta = g_ref(s1 + Q_phys w, d1) -
g_ref(s1, d1) (the change of AUC at d_ref caused by the physiology change alone; 0 when w = 0). Write to a separate file.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from scipy.stats import norm, spearmanr
sys.path.insert(0, os.getcwd())

LOG_FLAG = float(np.log(1.2))     # "clinically relevant" movement: AUC at the reference regimen changes by > 20 %
T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:7.1f}s] {msg}', flush=True)


# ---------------------------------------------------------------------------------------------- generic solver
def jacobian(resid, x):
    """R(x) [B, m] and its per-patient Jacobian [B, m, n]. Patient i's residuals depend on x_i only, so one backward
    pass per residual component (summed over patients) gives every patient's row at once."""
    x = x.detach().requires_grad_(True)
    R = resid(x)
    m = R.shape[1]
    J = torch.empty(R.shape[0], m, x.shape[1])
    for j in range(m):
        J[:, j] = torch.autograd.grad(R[:, j].sum(), x, retain_graph=j < m - 1)[0]
    return R.detach(), J


def lm(resid, x0, iters=40, tol=1e-4, tag=''):
    """Batched Levenberg-Marquardt (Marquardt scaling, Nielsen damping update) of 0.5 ||R_i(x_i)||^2 for every patient.
    Trial steps are scored in the same batched pass as the current point ([x; x + delta]), so the accept test is not
    disturbed by the shared adaptive step sequence."""
    x = x0.detach().clone()
    nb = x.shape[0]
    lam, nu = torch.full((nb,), 1e-2), torch.full((nb,), 2.0)
    for it in range(iters):
        R, J = jacobian(resid, x)
        A = J.transpose(1, 2) @ J
        g = (J.transpose(1, 2) @ R.unsqueeze(-1)).squeeze(-1)
        D = torch.diag_embed(torch.diagonal(A, dim1=1, dim2=2).clamp(min=1e-9))
        delta = -torch.linalg.solve(A + lam[:, None, None] * D, g.unsqueeze(-1)).squeeze(-1)
        pred_dec = -(delta * g).sum(1) - 0.5 * (delta.unsqueeze(1) @ A @ delta.unsqueeze(-1)).reshape(-1)
        with torch.no_grad():
            Rb = resid(torch.cat([x, x + delta]))
        c_old, c_new = 0.5 * (Rb[:nb] ** 2).sum(1), 0.5 * (Rb[nb:] ** 2).sum(1)
        rho = (c_old - c_new) / pred_dec.clamp(min=1e-12)
        acc = (rho > 0) & torch.isfinite(c_new)
        x = torch.where(acc[:, None], x + delta, x)
        lam = torch.where(acc, lam * torch.clamp(1 - (2 * rho - 1) ** 3, min=1 / 3), lam * nu).clamp(1e-9, 1e9)
        nu = torch.where(acc, torch.full_like(nu, 2.0), nu * 2)
        moved = torch.where(acc, delta.abs().amax(1), torch.zeros(nb))
        gmax = g.abs().amax(1)
        # converged: a tiny accepted step, a vanishing gradient, or a rejected step whose predicted gain is below what
        # the cost can resolve (float32 rounding and the shared adaptive step sequence, ~1e-5 in cost units)
        conv = ((moved < tol) & acc) | (gmax < tol) | (~acc & (pred_dec < 1e-5))
        if it % 5 == 0 or bool(conv.all()):
            log(f'  {tag} it {it:2d}: cost median {float(c_old.median()):.3f}, accepted {float(acc.float().mean()):.2f}, '
                f'converged {float(conv.float().mean()):.3f}, max step {float(moved.max()):.1e}')
        if it >= 3 and bool(conv.all()):
            break
    R, J = jacobian(resid, x)
    H = J.transpose(1, 2) @ J
    gmax = (J.transpose(1, 2) @ R.unsqueeze(-1)).squeeze(-1).abs().amax(1)
    return x.detach(), R, J, H, gmax


def log_evidence(R, J, H, m_data, sig):
    """Laplace log p(y | model): rows 0..m_data-1 of R are data residuals (y - f) / sig, the rest are whitened Gaussian
    prior residuals (linear in x, so their Jacobian rows give the prior precision exactly)."""
    Jp = J[:, m_data:]
    lam = Jp.transpose(1, 2) @ Jp
    return (-0.5 * (R ** 2).sum(1) - 0.5 * m_data * np.log(2 * np.pi * sig ** 2)
            + 0.5 * torch.logdet(lam) - 0.5 * torch.logdet(H))


def whitener(C):
    """W such that ||(s - m) @ W||^2 = (s - m) C^-1 (s - m)'  (row vectors, batched)."""
    n = C.shape[-1]
    jitter = 1e-9 * torch.diagonal(C, dim1=-2, dim2=-1).mean(-1)[..., None, None] * torch.eye(n)
    Lc = torch.linalg.cholesky(C + jitter)
    return torch.linalg.inv(Lc).transpose(-1, -2)


def wh(r, W):
    return (r.unsqueeze(1) @ W).squeeze(1)


def selftest():
    """Linear-Gaussian case: y = A x + e, x ~ N(m, P). The LM solution must equal the posterior mean, H^-1 the posterior
    covariance, and the Laplace evidence the exact N(y; A m, A P A' + s^2 I)."""
    torch.manual_seed(0)
    torch.set_default_dtype(torch.float64)
    B, n, md, s = 50, 4, 3, 0.1
    A = torch.randn(B, md, n)
    m = torch.randn(B, n)
    Lp = torch.randn(B, n, n) * 0.3 + 1.5 * torch.eye(n)
    P = Lp @ Lp.transpose(1, 2)
    W = whitener(P)
    x_true = m + (Lp @ torch.randn(B, n, 1)).squeeze(-1)
    y = (A @ x_true.unsqueeze(-1)).squeeze(-1) + s * torch.randn(B, md)
    resid = lambda x: torch.cat([(y.repeat(x.shape[0] // B, 1) - (A.repeat(x.shape[0] // B, 1, 1) @ x.unsqueeze(-1)).squeeze(-1)) / s,
                                 wh(x - m.repeat(x.shape[0] // B, 1), W.repeat(x.shape[0] // B, 1, 1))], 1)
    x, R, J, H, _ = lm(resid, torch.zeros(B, n), iters=30, tol=1e-10, tag='selftest')
    Pinv = torch.linalg.inv(P)
    post_cov = torch.linalg.inv(A.transpose(1, 2) @ A / s ** 2 + Pinv)
    post_mean = (post_cov @ (A.transpose(1, 2) @ y.unsqueeze(-1) / s ** 2 + Pinv @ m.unsqueeze(-1))).squeeze(-1)
    S = A @ P @ A.transpose(1, 2) + s ** 2 * torch.eye(md)
    exact = torch.distributions.MultivariateNormal((A @ m.unsqueeze(-1)).squeeze(-1), S).log_prob(y)
    lz = log_evidence(R, J, H, md, s)
    print(f'selftest: max |x - posterior mean| {float((x - post_mean).abs().max()):.2e}; '
          f'max |H^-1 - posterior cov| {float((torch.linalg.inv(H) - post_cov).abs().max()):.2e}; '
          f'max |log Z - exact| {float((lz - exact).abs().max()):.2e}')
    ok = (x - post_mean).abs().max() < 1e-6 and (lz - exact).abs().max() < 1e-6
    print('SELFTEST', 'OK' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


def ols(x, y):
    xc, yc = x - x.mean(), y - y.mean()
    b = float((xc * yc).sum() / (xc ** 2).sum())
    res = yc - b * xc
    se = float(np.sqrt((res ** 2).sum() / (len(x) - 2) / (xc ** 2).sum()))
    return b, se


def summarize(delta, x, sd=None):
    """delta: estimated log-change of the readout (truth 0); x = log(d2/d1)."""
    a = np.abs(delta)
    b, se = ols(x, delta)
    out = {'median_abs_pct': 100 * float(np.median(a)), 'p90_abs_pct': 100 * float(np.quantile(a, 0.9)),
           'mean_pct': 100 * float(np.mean(delta)), 'share_abs_gt20': float(np.mean(a > LOG_FLAG)),
           'slope_on_log_dose_ratio': b, 'slope_se': se, 'spearman_vs_log_dose_ratio': float(spearmanr(x, delta)[0])}
    if sd is not None:
        z = delta / sd
        p_big = norm.cdf((delta - LOG_FLAG) / sd) + norm.cdf((-LOG_FLAG - delta) / sd)
        out.update({'median_sd_pct': 100 * float(np.median(sd)), 'flag_z196': float(np.mean(np.abs(z) > 1.96)),
                    'flag_p_gt20_095': float(np.mean(p_big > 0.95)), 'z_sd': float(np.std(z))})
    return out


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kind', nargs='?', choices=['film', 'a', 'b'])
    ap.add_argument('ckpt', nargs='?')
    ap.add_argument('out', nargs='?')
    ap.add_argument('--cohort', default='confound_vc00_s4_pnoise_win')
    ap.add_argument('--scale-from', default='confound_vc00_s4_pnoise_win_n100')
    ap.add_argument('--n', type=int, default=0, help='first n test patients (0 = all)')
    ap.add_argument('--q', type=float, default=0.5, help='Jq: sd of the between-visit move in units of the prior sd')
    ap.add_argument('--dref', type=float, default=4.0, help='reference dose of the readout, mg')
    ap.add_argument('--sigmas', default='eb,ckpt', help="emission sds to run: 'eb', 'ckpt' or numbers, comma-separated")
    ap.add_argument('--n-eb', type=int, default=300, help='test patients whose V1 samples estimate sigma (eb)')
    ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--phys-only', action='store_true', help='kind a: track the physiology block only (see docstring)')
    cli = ap.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    if cli.selftest:
        selftest()
    if not (cli.kind and cli.ckpt and cli.out):
        ap.error('kind, ckpt and out are required')

    from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film, set_static_hematocrit
    from probe_entanglement import load_film_model
    from test_dose_cond import unscale_data, calculate_auc

    dev = torch.device('cpu')
    a0 = torch.load(cli.ckpt, map_location='cpu', weights_only=False)['args']
    set_static_hematocrit(int(getattr(a0, 'static_dim', 3)) >= 4)
    model, args = load_film_model(cli.ckpt, dev)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    is_b = getattr(model, 'dose_hist', None) is not None
    if is_b != (cli.kind == 'b') or (getattr(model, 'orth_split', None) is not None) != (cli.kind in ('a', 'b')):
        sys.exit(f'kind {cli.kind} does not match the checkpoint')
    if getattr(model, 'film_time', False) or getattr(model, 'n_film_occ', 0):
        sys.exit('film_time / film_occ checkpoints are not supported')
    if cli.kind != 'b' and not getattr(model, 'film_no_z0_cond', False):
        sys.exit('the prior push-forward assumes an affine FiLM transport (--film-no-z0-cond)')
    if cli.phys_only and cli.kind != 'a':
        sys.exit('--phys-only is for kind a (FiLM + orth-split)')

    # ---------------- data (scaler pinned to the training cohort, as test_film_matched --scale-from) ----------------
    pin = extract_gen_tac_film(file_path=[f'results/exp_film_run/{cli.scale_from}/virtual_cohort_film_train.csv'])[1]
    MAX_OUT, LAM, DMAX = float(pin[0]), float(pin[1]), float(pin[2])

    def batch_of(d, keys):
        ds = TacroFilmDataset({k: d[k] for k in keys})
        return collate_fn_tacro_film([ds[i] for i in range(len(keys))], dev)

    tr = f'results/exp_film_run/{cli.cohort}/virtual_cohort_film_train.csv'
    te = f'results/exp_film_run/{cli.cohort}/virtual_cohort_film_test.csv'
    data, scaler = extract_gen_tac_film(file_path=[tr, te], scale=pin)
    train_ids = set(extract_gen_tac_film(file_path=[tr], scale=pin)[0].keys())
    ids = [k for k in data if k not in train_ids]
    if cli.n:
        ids = ids[:cli.n]
    bt = batch_of(data, ids)
    dtrain = extract_gen_tac_film(file_path=[f'results/exp_film_run/{cli.scale_from}/virtual_cohort_film_train.csv'],
                                  scale=pin)[0]
    btr = batch_of(dtrain, list(dtrain.keys()))
    NB = len(ids)
    Y1, Y2 = bt['observed_data_v1'][..., 0], bt['observed_data_v2'][..., 0]
    D1, D2 = bt['dose_v1'].float(), bt['dose_v2'].float()
    FM = bt['static_v1'][:, 1].float()
    TRUE1 = (bt['auc_red_v1'] * MAX_OUT).numpy().astype(float)
    TRUE2 = (bt['auc_red_v2'] * MAX_OUT).numpy().astype(float)
    XLR = np.log((D2 / D1).numpy().astype(float))
    TP_OBS = bt['observed_tp_v1'].float()
    TP_D = torch.linspace(0, 24, 100)
    SEG12 = ((TP_D[1:] <= 12.0) & (TP_D[:-1] <= 12.0)).float()
    DREF = cli.dref / DMAX
    log(f'{cli.kind}: {cli.ckpt}\n  {NB} test patients of {cli.cohort}, scaler of {cli.scale_from} (max_out {MAX_OUT:.2f}, '
        f'lambda {LAM:.4f}, dose max {DMAX:g} mg); {len(dtrain)} training patients for the prior; sample times '
        f'{TP_OBS.tolist()} h; |log d2/d1| median {np.median(np.abs(XLR)):.2f}')

    # ---------------- the frozen model as an emission ----------------
    L = int(args.latents)
    if cli.kind == 'b':
        kfd = model.orth_split.k_f + model.orth_split.k_d
        QP = model.orth_split.Q()[:, kfd:].detach().clone()      # [L, n_p] physiology basis
        NS = QP.shape[1]
    else:
        NS = L
    if cli.kind == 'a':
        PROJ = {k: P().detach().clone() for k, P in (('film', model.orth_split.P_film), ('dyn', model.orth_split.P_dyn),
                                                      ('phys', model.orth_split.P_phys))}

    def predict(s, dose, tp, form):
        """state [B, NS] at the visit's dose -> prediction [B, T] in model units."""
        if cli.kind == 'b':
            sol = model._dose_history_traj((s @ QP.t()).unsqueeze(0), dose, tp, form=form)
        else:
            sol = model.diffeq_solver(s.unsqueeze(0), tp)
        return model.decoder(sol)[0, :, :, 0]

    def transport(s, d_from, d_to):
        """The model's own dose transport (FiLM / A); identity for B (its state is dose-free)."""
        if cli.kind == 'b':
            return s
        z = s.unsqueeze(0)
        c = model._film_context(torch.stack([d_from, d_to], 1).unsqueeze(0), z.detach(), None)
        gamma, beta = model._film_maps(c, d_from, d_to, 1)
        return model._film_apply(z, gamma, beta)[0]

    def affine(d_from, d_to):
        """T(z) = z @ A + b per patient (row convention), A [B, NS, NS], b [B, NS]."""
        nb = d_from.shape[0]
        b = transport(torch.zeros(nb, NS), d_from, d_to)
        E = torch.eye(NS)
        return torch.stack([transport(E[j].expand(nb, NS), d_from, d_to) - b for j in range(NS)], 1), b

    def auc(pred, form):
        """Window AUC (ng.h/mL) exactly as test_dose_cond.calculate_auc: inverse Box-Cox (NaN -> 0) * max_out,
        trapezoid to 12 h for Prograf, 24 h for Advagraf."""
        base = LAM * pred + 1.0
        conc = torch.where(base > 0, base.clamp(min=1e-12) ** (1.0 / LAM), torch.zeros_like(base)) * MAX_OUT
        tr_ = 0.5 * (conc[:, 1:] + conc[:, :-1]) * (TP_D[1:] - TP_D[:-1])
        return torch.where(form > 0.5, (tr_ * SEG12).sum(1), tr_.sum(1))

    def g_ref(s, d_visit, form):
        dref = torch.full_like(d_visit, DREF)
        z = s if cli.kind == 'b' else transport(s, d_visit, dref)
        return torch.log(auc(predict(z, dref, TP_D, form), form))

    def auc_at(s, dose, form):
        with torch.no_grad():
            return auc(predict(s, dose, TP_D, form), form).numpy().astype(float)

    def tile(t, k):
        return t if k == 1 else torch.cat([t] * k)

    # ---------------- encoder + population prior ----------------
    def encode(b, v):
        enc_in, st = model._encoder_input(b[f'observed_data_{v}'], b[f'dose_{v}'], b[f'static_{v}'])
        mu, sd = model.encoder_z0(enc_in, b[f'observed_tp_{v}'], static=st, run_backwards=True)
        mu, var = mu[0], sd[0].abs() ** 2
        if cli.kind == 'b':
            return mu @ QP, torch.einsum('li,bl,lj->bij', QP, var, QP)
        return mu, torch.diag_embed(var)

    with torch.no_grad():
        pm1, pc1 = encode(btr, 'v1')
        pm2, pc2 = encode(btr, 'v2')
        PMU, PCV = torch.cat([pm1, pm2]), torch.cat([pc1, pc2])
        PDOSE = torch.cat([btr['dose_v1'], btr['dose_v2']]).float()
        if cli.kind == 'b':
            M_REF, P_REF = PMU.mean(0), torch.cov(PMU.T) + PCV.mean(0)
        else:
            A_, b_ = affine(PDOSE, torch.full_like(PDOSE, DREF))
            ZR = (PMU.unsqueeze(1) @ A_).squeeze(1) + b_
            M_REF, P_REF = ZR.mean(0), torch.cov(ZR.T) + (A_.transpose(1, 2) @ PCV @ A_).mean(0)
        ev = torch.linalg.eigvalsh(P_REF)

    def prior_at(d):
        nb = d.shape[0]
        if cli.kind == 'b':
            return M_REF.expand(nb, NS).clone(), P_REF.expand(nb, NS, NS).clone()
        A_, b_ = affine(torch.full_like(d, DREF), d)
        return (M_REF.expand(nb, NS).unsqueeze(1) @ A_).squeeze(1) + b_, A_.transpose(1, 2) @ P_REF @ A_

    with torch.no_grad():
        M1, C1 = prior_at(D1)
        M2, C2 = prior_at(D2)
        W1, W2 = whitener(C1), whitener(C2)
        E1m, E1c = encode(bt, 'v1')
        E2m, E2c = encode(bt, 'v2')
    log(f'prior: tr P_ref {float(ev.sum()):.2f}, eigenvalues {np.round(ev.numpy()[::-1], 3).tolist()}')

    # ---------------- residuals ----------------
    def resid_ind(Y, D, F, M, W, sig):
        def r(x):
            k = x.shape[0] // Y.shape[0]
            return torch.cat([(tile(Y, k) - predict(x, tile(D, k), TP_OBS, tile(F, k))) / sig, wh(x - tile(M, k), tile(W, k))], 1)
        return r

    def resid_joint(mode, sig, WQ=None):
        def r(x):
            k = x.shape[0] // NB
            y1, y2, d1, d2, fm, m1, w1 = (tile(t, k) for t in (Y1, Y2, D1, D2, FM, M1, W1))
            if mode == 'j0':
                s1 = x
                s2 = transport(s1, d1, d2)
            else:
                s1, s2 = x[:, :NS], x[:, NS:]
            p = predict(torch.cat([s1, s2]), torch.cat([d1, d2]), TP_OBS, torch.cat([fm, fm]))
            nb = x.shape[0]
            parts = [(y1 - p[:nb]) / sig, (y2 - p[nb:]) / sig, wh(s1 - m1, w1)]
            if mode == 'jq':
                parts.append(wh(s2 - transport(s1, d1, d2), tile(WQ, k)))
            return torch.cat(parts, 1)
        return r

    # ---------------- sigma ----------------
    sig_ckpt = float(getattr(args, 'obsrv_std', 0.05) or 0.05)
    sigmas = []
    for tok in cli.sigmas.split(','):
        if tok == 'ckpt':
            sigmas.append(('ckpt', sig_ckpt))
        elif tok == 'eb':
            ne = min(cli.n_eb, NB)
            grid = [0.03, 0.04, 0.05, 0.065, 0.08, 0.1, 0.125, 0.15, 0.2]
            lzs = []
            for sg in grid:
                x, R, J, H, _ = lm(resid_ind(Y1[:ne], D1[:ne], FM[:ne], M1[:ne], W1[:ne], sg), E1m[:ne], iters=25,
                                   tag=f'eb {sg:g}')
                lzs.append(float(log_evidence(R, J, H, 3, sg).sum()))
            i = int(np.argmax(lzs))
            if 0 < i < len(grid) - 1:      # parabola in log sigma through the best grid point and its neighbours
                xs, ys = np.log(grid[i - 1:i + 2]), np.array(lzs[i - 1:i + 2])
                c2, c1, _ = np.polyfit(xs, ys, 2)
                s_eb = float(np.exp(-c1 / (2 * c2))) if c2 < 0 else grid[i]
            else:
                s_eb = grid[i]
            log(f'sigma eb: log evidence on {ne} V1 visits {dict(zip(grid, np.round(lzs, 1)))} -> {s_eb:.4f} '
                f'(checkpoint obsrv_std {sig_ckpt:g})')
            sigmas.append(('eb', s_eb))
            eb_curve = dict(zip([str(g) for g in grid], lzs))
        else:
            sigmas.append((tok, float(tok)))

    # ---------------- readouts ----------------
    def grad_g(s, d, form):
        s = s.detach().requires_grad_(True)
        gv = g_ref(s, d, form)
        return gv.detach().numpy().astype(float), torch.autograd.grad(gv.sum(), s)[0]

    def quad(gr, C):   # gr [B, n], C [B, n, n] -> gr' C gr
        return (gr.unsqueeze(1) @ C @ gr.unsqueeze(-1)).reshape(-1).numpy().astype(float)

    def disp_stats(s1, s2):
        with torch.no_grad():
            d = s2 - transport(s1, D1, D2)
        out = {'identity_ratio': ((d ** 2).sum(1) / (2 * float(ev.sum()))).numpy().astype(float).tolist()}
        if cli.kind == 'a':
            tot = (d ** 2).sum(1).clamp(min=1e-12)
            for k, P in PROJ.items():
                out[f'frac_{k}'] = (((d @ P) ** 2).sum(1) / tot).numpy().astype(float).tolist()
        return out

    def fit_stats(s1, s2):
        cf = auc_at(transport(s1, D1, D2) if cli.kind != 'b' else s1, D2, FM)
        return {'auc_fit_v1': auc_at(s1, D1, FM).tolist(), 'auc_fit_v2': auc_at(s2, D2, FM).tolist(),
                'auc_cf_v2_from_v1': cf.tolist()}

    res = {'kind': cli.kind, 'ckpt': cli.ckpt, 'cohort': cli.cohort, 'scale_from': cli.scale_from, 'n': NB,
           'dref_mg': cli.dref, 'q': cli.q, 'prior_trace': float(ev.sum()), 'state_dims': NS,
           'ids': [str(k) for k in ids], 'dose_v1_mg': (D1 * DMAX).numpy().astype(float).tolist(),
           'dose_v2_mg': (D2 * DMAX).numpy().astype(float).tolist(), 'prograf': FM.numpy().astype(int).tolist(),
           'true_auc_v1': TRUE1.tolist(), 'true_auc_v2': TRUE2.tolist(), 'methods': {}}
    if 'eb_curve' in locals():
        res['sigma_eb_curve'] = eb_curve

    # sanity: the torch AUC reproduces test_dose_cond.calculate_auc on the encoder-mean curves
    with torch.no_grad():
        pr = predict(E1m[:50], D1[:50], TP_D, FM[:50])
        ref = calculate_auc(unscale_data(pr, scaler), TP_D.numpy(), bt['static_v1'][:50])
        mine = auc(pr, FM[:50]).numpy()
    log(f'check: torch AUC vs calculate_auc max rel diff {float(np.max(np.abs(mine / ref - 1))):.1e}')

    def run_method(name, s1, s2, sd=None, extra=None, delta=None):
        g1, _ = grad_g(s1, D1, FM)
        g2, _ = grad_g(s2, D2, FM)
        if delta is None:
            delta = g2 - g1
        m = {'summary': summarize(delta, XLR, sd), 'delta': delta.tolist(), 'g1': g1.tolist(), 'g2': g2.tolist()}
        if sd is not None:
            m['sd'] = sd.tolist()
        m.update(fit_stats(s1, s2))
        m.update(disp_stats(s1, s2))
        for k in ('auc_fit_v1', 'auc_fit_v2', 'auc_cf_v2_from_v1'):
            t = TRUE1 if k == 'auc_fit_v1' else TRUE2
            m['summary'][f'{k}_median_abs_err_pct'] = 100 * float(np.median(np.abs(np.array(m[k]) / t - 1)))
        m['summary']['identity_ratio_median'] = float(np.median(m['identity_ratio']))
        if cli.kind == 'a':
            for k in PROJ:
                m['summary'][f'disp_frac_{k}_median'] = float(np.median(m[f'frac_{k}']))
        if extra:
            m.update(extra)
            m['summary'].update({k: v for k, v in extra.items() if not isinstance(v, list)})
        res['methods'][name] = m
        s = m['summary']
        log(f'{name:14s} |Delta| median {s["median_abs_pct"]:5.1f}% p90 {s["p90_abs_pct"]:5.1f}% mean {s["mean_pct"]:+5.1f}% '
            f'slope {s["slope_on_log_dose_ratio"]:+.3f} (se {s["slope_se"]:.3f})'
            + (f' flags z {100 * s["flag_z196"]:.1f}% >20% {100 * s["flag_p_gt20_095"]:.1f}% sd~{s["median_sd_pct"]:.1f}%'
               if sd is not None else '')
            + f' | fit V1 {s["auc_fit_v1_median_abs_err_pct"]:.1f}% V2 {s["auc_fit_v2_median_abs_err_pct"]:.1f}% '
              f'cf V2 {s["auc_cf_v2_from_v1_median_abs_err_pct"]:.1f}% | id ratio {s["identity_ratio_median"]:.3f}')

    # --phys-only: A's physiology block (one shared non-physiology state; between visits only a physiology change w)
    if cli.phys_only:
        kfd = model.orth_split.k_f + model.orth_split.k_d
        QPA = model.orth_split.Q()[:, kfd:].detach().clone()      # [L, n_p] physiology basis (p = z @ QPA)
        NPH = QPA.shape[1]
        with torch.no_grad():
            MP, CP = M1 @ QPA, QPA.t() @ C1 @ QPA                  # physiology marginal of the prior at d1
            gap = max(float((M2 @ QPA - MP).abs().max()), float((QPA.t() @ C2 @ QPA - CP).abs().max()))
        log(f'phys-only: {NPH} physiology dims; physiology prior at d1 vs d2 max |diff| {gap:.1e} (dose-free if ~0)')
        res.update({'phys_only': True, 'phys_dims': NPH, 'phys_prior_dose_gap': gap})

        def resid_phys(mode, sig, WW):
            def r(x):
                k = x.shape[0] // NB
                y1, y2, d1, d2, fm, m1, w1, ww, mp = (tile(t, k) for t in (Y1, Y2, D1, D2, FM, M1, W1, WW, MP))
                s1, w = x[:, :NS], x[:, NS:]
                s2 = transport(s1 + w @ QPA.t(), d1, d2)
                p = predict(torch.cat([s1, s2]), torch.cat([d1, d2]), TP_OBS, torch.cat([fm, fm]))
                nb = x.shape[0]
                parts = [(y1 - p[:nb]) / sig, (y2 - p[nb:]) / sig, wh(s1 - m1, w1),
                         wh(w, ww) if mode == 'jpq' else wh(s1 @ QPA + w - mp, ww)]
                return torch.cat(parts, 1)
            return r

        def phys_deltas(x):
            """[(Delta, dDelta/dx)] for the path-inclusive and the clean readout, x = [s1, w]."""
            out = []
            for clean in (False, True):
                xx = x.detach().requires_grad_(True)
                s1, w = xx[:, :NS], xx[:, NS:]
                moved = s1 + w @ QPA.t()
                dv = (g_ref(moved, D1, FM) if clean else g_ref(transport(moved, D1, D2), D2, FM)) - g_ref(s1, D1, FM)
                out.append((dv.detach().numpy().astype(float), torch.autograd.grad(dv.sum(), xx)[0]))
            return out

        def run_phys(sig, tag, xj0, lz0):
            for mode, WW in (('indp', whitener(CP)), (f'jpq{cli.q:g}', whitener(cli.q ** 2 * CP))):
                x0 = torch.cat([xj0, torch.zeros(NB, NPH)], 1)
                x, R, J, H, gm = lm(resid_phys('indp' if mode == 'indp' else 'jpq', sig, WW), x0, iters=cli.iters,
                                    tag=f'{mode}{tag}')
                lbf = (log_evidence(R, J, H, 6, sig) - lz0).numpy().astype(float)
                Hinv = torch.linalg.inv(H)
                s1 = x[:, :NS]
                with torch.no_grad():
                    s2 = transport(s1 + x[:, NS:] @ QPA.t(), D1, D2)
                    cons = float(((s2 - s1) @ QPA - x[:, NS:]).abs().max())
                ext = {'sigma': sig, 'grad_max_median': float(gm.median()), 'log_bf_vs_j0': lbf.tolist(),
                       'bf_gt3': float(np.mean(lbf > np.log(3))), 'bf_gt10': float(np.mean(lbf > np.log(10))),
                       'log_bf_median': float(np.median(lbf)), 'rms_resid_sigma_units': float(R[:, :6].pow(2).mean().sqrt()),
                       'w_norm_median': float(x[:, NS:].norm(dim=1).median()), 'phys_conservation_max': cons}
                for suffix, (dv, gr) in zip(('', 'c'), phys_deltas(x)):
                    run_method(mode + suffix + tag, s1, s2, sd=np.sqrt(quad(gr, Hinv)), extra=dict(ext), delta=dv)
                log(f'   {mode} vs J0: log BF median {np.median(lbf):+.2f}, BF > 3 for {100 * np.mean(lbf > np.log(3)):.1f}%, '
                    f'> 10 for {100 * np.mean(lbf > np.log(10)):.1f}%; |w| median {ext["w_norm_median"]:.3f}; '
                    f'max |p2 - p1 - w| {cons:.1e}')

    # ENC: amortised posterior (sd from the encoder's own covariance)
    _, gr1 = grad_g(E1m, D1, FM)
    _, gr2 = grad_g(E2m, D2, FM)
    run_method('enc', E1m, E2m, sd=np.sqrt(quad(gr1, E1c) + quad(gr2, E2c)))

    for slab, sig in sigmas:
        tag = f'_{slab}'
        log(f'=== sigma {slab} = {sig:.4f}')
        if not cli.phys_only:
            x1, R1, J1, H1, gm1 = lm(resid_ind(Y1, D1, FM, M1, W1, sig), E1m, iters=cli.iters, tag=f'ind1{tag}')
            x2, R2, J2, H2, gm2 = lm(resid_ind(Y2, D2, FM, M2, W2, sig), E2m, iters=cli.iters, tag=f'ind2{tag}')
            _, gr1 = grad_g(x1, D1, FM)
            _, gr2 = grad_g(x2, D2, FM)
            sd = np.sqrt(quad(gr1, torch.linalg.inv(H1)) + quad(gr2, torch.linalg.inv(H2)))
            run_method('ind' + tag, x1, x2, sd=sd, extra={
                'sigma': sig, 'grad_max_median': float(torch.cat([gm1, gm2]).median()),
                'grad_max_max': float(torch.cat([gm1, gm2]).max()),
                'rms_resid_v1_sigma_units': float(R1[:, :3].pow(2).mean().sqrt()),
                'rms_resid_v2_sigma_units': float(R2[:, :3].pow(2).mean().sqrt())})

        xj0, Rj0, Jj0, Hj0, gm0 = lm(resid_joint('j0', sig), E1m, iters=cli.iters, tag=f'j0{tag}')
        lz0 = log_evidence(Rj0, Jj0, Hj0, 6, sig)
        with torch.no_grad():
            s2j0 = transport(xj0, D1, D2)
        run_method('j0' + tag, xj0, s2j0, extra={'sigma': sig, 'grad_max_median': float(gm0.median()),
                                                 'rms_resid_sigma_units': float(Rj0[:, :6].pow(2).mean().sqrt())})
        if cli.phys_only:
            run_phys(sig, tag, xj0, lz0)
            continue

        WQ = whitener(cli.q ** 2 * C2)
        xq, Rq, Jq, Hq, gmq = lm(resid_joint('jq', sig, WQ), torch.cat([xj0, s2j0], 1), iters=cli.iters, tag=f'jq{tag}')
        lzq = log_evidence(Rq, Jq, Hq, 6, sig)
        s1q, s2q = xq[:, :NS], xq[:, NS:]
        _, gq1 = grad_g(s1q, D1, FM)
        _, gq2 = grad_g(s2q, D2, FM)
        gd = torch.cat([-gq1, gq2], 1)
        sdq = np.sqrt(quad(gd, torch.linalg.inv(Hq)))
        lbf = (lzq - lz0).numpy().astype(float)
        run_method(f'jq{cli.q:g}' + tag, s1q, s2q, sd=sdq, extra={
            'sigma': sig, 'grad_max_median': float(gmq.median()), 'log_bf_vs_j0': lbf.tolist(),
            'bf_gt3': float(np.mean(lbf > np.log(3))), 'bf_gt10': float(np.mean(lbf > np.log(10))),
            'log_bf_median': float(np.median(lbf)),
            'rms_resid_sigma_units': float(Rq[:, :6].pow(2).mean().sqrt())})
        log(f'   Jq vs J0: log BF median {np.median(lbf):+.2f}, BF > 3 for {100 * np.mean(lbf > np.log(3)):.1f}%, '
            f'> 10 for {100 * np.mean(lbf > np.log(10)):.1f}% (truth: no movement)')

    os.makedirs(os.path.dirname(cli.out) or '.', exist_ok=True)
    with open(cli.out, 'w') as f:
        json.dump(res, f)
    log(f'wrote {cli.out}')


if __name__ == '__main__':
    main()
