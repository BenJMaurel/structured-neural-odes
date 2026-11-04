"""Amortisation-gap check for the EB-VAE on scenario 3 (PharmaNODE 2026-10-08).
The trained model is FROZEN; for each patient its latent theta (the posterior-mean space, dim IC + DYN) is re-fitted by MAP on
the SAME samples the encoder sees (ENC-flagged visit-1 0/1/3 h):
    theta* = argmin  sum_obs [ 0.5 ((y - f(theta))/sd)^2 + log sd ]  +  0.5 (theta - mu_p)' Sigma_p^-1 (theta - mu_p)
with the model's own learned prior N(mu_p(cov, d1), Sigma_p) and sd = the model's own learned noise ('model') or the
simulator's true sd 0.71 + 0.113 f ('true'). Start at the encoder's posterior mean; Adam, fixed iterations. Then the factual
AUC exactly as run_scen_baaz.py evaluate(). If MAP on new patients approaches the encoder's TRAINING-patient error, the
encoder (amortised inference) is what fails to generalise; if MAP is no better, the decoder / prior are.
    map_refine.py <models.pt> <train.csv> <driver test CSVs, comma-separated> <eval.csv> <out.json> [--iters 300] [--lr 0.05]
Checks: theta = encoder mean reproduces the encoder's k exactly, and its AUCs reproduce eval_saved.py."""
import os, sys, json, argparse
os.environ.update(BAAZ_PER_DOSE='1', BAAZ_COV_COLS='FORM,CYP,HTN', BAAZ_ENC_COL='ENC')
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from lib.utils.utils_preprocess import TrajectoryDataset, make_collate_fn, compute_global_stats, destandardize_concentration
from lib.utils.utils_shared import preprocess_batch, encode_latent, prepare_ode_input, make_predictions, normalize_encoder_input
from lib.models.NNmodels import Encoder_Transformer, ODEFunc, SimpleDecoder, TrainableNoise

ap = argparse.ArgumentParser()
ap.add_argument('ckpt'); ap.add_argument('train'); ap.add_argument('tests'); ap.add_argument('eval'); ap.add_argument('out')
ap.add_argument('--iters', type=int, default=300); ap.add_argument('--lr', type=float, default=0.05)
cli = ap.parse_args()
IC, DYN, LAT = (int(x) for x in os.environ.get('BAAZ_DIMS', '1,2,2').split(','))
dev = torch.device('cpu'); torch.set_num_threads(1); torch.manual_seed(0)
df = pd.read_csv(cli.train, sep=';'); gmd, gmt, gm, gs, gmax, gmin = compute_global_stats(df)
kw = dict(max_dose=gmd, max_time=gmt, max_value=gmax, min_value=gmin, global_mean=gm, global_std=gs)
tests = cli.tests.split(',')
dose_t = np.unique(np.concatenate([pd.read_csv(f, sep=';').query('EVID > 0').TIME.values for f in [cli.train] + tests])) / gmt
n_grid = int(round(gmt / 0.25)) + 1
t_dense = torch.tensor(np.unique(np.concatenate([np.linspace(0, 1, n_grid), dose_t]).astype(np.float32)), dtype=torch.float32)


def build(h_enc, m_enc, layers, add, learn_cov):
    enc = Encoder_Transformer(dim_latent=LAT, dim_parameters_IC=IC, dim_parameters_dynamic=DYN, dim_cov=3, model_dim=m_enc, hidden_dim=h_enc,
                              num_heads=2, num_layers=layers, dropout=0.1 if layers == 1 else 0.05, cov_diag_epsilon=1e-5,
                              learn_prior_mean=True, learn_prior_covariance=learn_cov, diagonal_only=False, IC_dose_dependent=False)
    f = ODEFunc(dim_latent=LAT, dim_parameters_IC=IC, dim_parameters_dynamic=DYN, hid_dim=32, drug_dim=1)
    return {'func': f, 'encoder': enc, 'reducer': SimpleDecoder(LAT), 'noise': TrainableNoise(init_add_std=add, init_prop_std=0.01)}


S = torch.load(cli.ckpt, map_location='cpu', weights_only=False)
med, eb = build(8, 16, 1, 0.2, False), build(16, 32, 2, 0.1, True)
for M, key in ((med, 'med'), (eb, 'eb')):
    for k, m in M.items():
        m.load_state_dict(S[key][k]); m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
E = eb['encoder']


def theta_to_k(theta):
    """the encoder's own map from the posterior-mean space to the ODE initial state + dynamic parameters (AE mode)"""
    z0 = E.z0_mu.repeat(theta.shape[0], 1)
    if E.dim_parameter_IC > 0:
        z0 = E.z0_transformed(torch.cat([z0, theta[:, :E.dim_parameter_IC]], dim=-1))
    return torch.cat([z0, theta[:, E.dim_parameter_IC:]], dim=-1)


dte = TrajectoryDataset(cli.eval, **kw); raw = pd.read_csv(cli.eval, sep=';')
obs = raw[raw.EVID == 0]; info = obs.groupby('ID').agg(form=('FORM', 'first')); auc = obs.groupby('ID').AUC.first()
g24 = np.round(np.arange(0, 24 + 1e-9, 0.05), 4)
res = {m: dict(id=[], true_auc=[], pred_auc=[]) for m in ('encoder', 'map_model_noise', 'map_true_noise')}
diag = dict(obj_start={}, obj_end={}, rel_drop={}, step_last={})


def aucs(k, x_pad, cov, dose, dts, evid, idl):
    with torch.no_grad():
        f = prepare_ode_input(x_pad, k, eb['func'], cov, dose, dts, evid, enable_ae=True, enable_onlymedian=False)
        tg = torch.tensor(np.tile(g24 / gmt, (len(idl), 1)), dtype=torch.float32)
        pred = make_predictions(tg, t_dense, k, f, eb['reducer'], gm, gs)[0].numpy()
    out = []
    for j, i in enumerate(idl):
        tau = 12.0 if info.loc[i].form == 1 else 24.0; m = g24 <= tau + 1e-9
        out.append(float(np.trapezoid(pred[j, m], g24[m])))
    return out


for batch in DataLoader(dte, batch_size=50, shuffle=False, collate_fn=make_collate_fn(dte.global_max_len)):
    ids, tr, t_pad, x_pad, t_enc, x_enc, t_cut, x_cut, m_enc, cov, dose, dts, evid = preprocess_batch(batch, dev, truncation=1)
    idl = ids.tolist()
    with torch.no_grad():
        x_in = normalize_encoder_input(x_pad, t_pad, x_pad, cov, dose, m_enc, dts, evid, med['encoder'], med['func'], med['reducer'], t_dense, gm, gs)
        k_enc, z0, mu_q, L_q, mu_p, L_p, mL1, rf = encode_latent(E, t_pad, x_in, cov, dose, m_enc, enable_vae=False, enable_ae=True, enable_onlymedian=False)
        assert torch.allclose(theta_to_k(mu_q), k_enc, atol=1e-6), 'theta -> k map does not reproduce the encoder'
    Mobs = pad_sequence([m.bool() for m in m_enc], batch_first=True)
    Mobs = torch.nn.functional.pad(Mobs, (0, t_pad.shape[1] - Mobs.shape[1])).float()
    y = destandardize_concentration(x_pad, gm, gs)
    Lp = L_p if L_p.dim() == 3 else L_p.expand(len(idl), *L_p.shape)
    for tag, noise in (('encoder', None), ('map_model_noise', 'model'), ('map_true_noise', 'true')):
        if noise is None:
            th = mu_q.detach()
        else:
            th = mu_q.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([th], lr=cli.lr)

            def objective(th):
                k = theta_to_k(th)
                f = prepare_ode_input(x_pad, k, eb['func'], cov, dose, dts, evid, enable_ae=True, enable_onlymedian=False)
                pred = make_predictions(t_pad, t_dense, k, f, eb['reducer'], gm, gs)[0]
                sd = eb['noise']._compute_sigma_total(pred) if noise == 'model' else 0.71 + 0.113 * pred.abs()
                nll = ((0.5 * ((y - pred) / sd) ** 2 + torch.log(sd)) * Mobs).sum(1)
                r = torch.linalg.solve_triangular(Lp, (th - mu_p).unsqueeze(-1), upper=False).squeeze(-1)
                return nll + 0.5 * (r ** 2).sum(1)                      # per patient

            o0 = objective(th).detach()
            for it in range(cli.iters):
                g = opt.param_groups[0]; g['lr'] = cli.lr * 0.5 * (1 + np.cos(np.pi * it / cli.iters))   # cosine decay
                prev = th.detach().clone()
                opt.zero_grad(); objective(th).sum().backward(); opt.step()
            o1 = objective(th).detach()
            for j, i in enumerate(idl):
                diag['obj_start'].setdefault(tag, []).append(float(o0[j])); diag['obj_end'].setdefault(tag, []).append(float(o1[j]))
                diag['step_last'].setdefault(tag, []).append(float((th.detach() - prev)[j].abs().max()))
            th = th.detach()
        a = aucs(theta_to_k(th), x_pad, cov, dose, dts, evid, idl)
        for j, i in enumerate(idl):
            res[tag]['id'].append(int(i)); res[tag]['true_auc'].append(float(auc.loc[i])); res[tag]['pred_auc'].append(a[j])

summ = {}
for tag, r in res.items():
    e = np.array(r['pred_auc']) / np.array(r['true_auc']) - 1
    summ[tag] = dict(median_abs=100 * float(np.median(np.abs(e))), rmspe=100 * float(np.sqrt(np.mean(e ** 2))), bias=100 * float(np.median(e)), n=len(e))
for tag in ('map_model_noise', 'map_true_noise'):
    o0, o1 = np.array(diag['obj_start'][tag]), np.array(diag['obj_end'][tag])
    summ[tag].update(obj_drop_median=float(np.median(o0 - o1)), obj_increased=int((o1 > o0 + 1e-6).sum()),
                     last_step_max=float(np.max(diag['step_last'][tag])))
json.dump(dict(ckpt=cli.ckpt, eval=cli.eval, dims=[IC, DYN, LAT], iters=cli.iters, lr=cli.lr, per_patient=res, summary=summ, diag=diag), open(cli.out, 'w'))
print(f'{os.path.basename(cli.ckpt)} on {os.path.basename(cli.eval)}: ' + '; '.join(
    f'{t} med {s["median_abs"]:.1f} RMSPE {s["rmspe"]:.1f}' + (f' (obj drop {s["obj_drop_median"]:.2f}, up {s["obj_increased"]}, last step {s["last_step_max"]:.1e})' if 'obj_drop_median' in s else '')
    for t, s in summ.items()), flush=True)
