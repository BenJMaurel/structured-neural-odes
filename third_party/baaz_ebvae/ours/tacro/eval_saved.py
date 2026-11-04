"""Evaluation-only re-run of a SAVED EB-VAE checkpoint (PharmaNODE 2026-10-08):
is the EB-VAE's error on our tacrolimus benchmarks a capacity limit (training patients fitted as badly as test patients) or a
generalisation gap (training patients fitted well)? Rebuilds the two-stage models exactly as the driver did (same dims, same
normalisation statistics from the training CSV, same dense grid from the training CSV + the driver's test CSVs), loads the
state dicts and scores ANY CSV with the driver's own prediction rule: encoder on the ENC-flagged samples (visit-1 0/1/3 h),
posterior mean, ODE through the patient's own dosing timeline.
    kind part2: run_tacro_baaz_curve.py -> V1 and V2 AUC;  kind scen: run_scen_baaz.py -> factual AUC of the observed interval.
    eval_saved.py <part2|scen> <models.pt> <train.csv> <driver test CSVs, comma-separated> <eval.csv> <out.json>
env BAAZ_DIMS="IC,DYN,LATENT" (default 1,2,2 = the published configuration) must match the checkpoint."""
import os, sys, json
os.environ.update(BAAZ_PER_DOSE='1', BAAZ_COV_COLS='FORM,CYP,HTN', BAAZ_ENC_COL='ENC')
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
from torch.utils.data import DataLoader
from lib.utils.utils_preprocess import TrajectoryDataset, make_collate_fn, compute_global_stats
from lib.utils.utils_shared import preprocess_batch, encode_latent, prepare_ode_input, make_predictions, normalize_encoder_input
from lib.models.NNmodels import Encoder_Transformer, ODEFunc, SimpleDecoder, TrainableNoise

kind, ckpt, P_tr, tests, P_ev, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(','), sys.argv[5], sys.argv[6]
IC, DYN, LAT = (int(x) for x in os.environ.get('BAAZ_DIMS', '1,2,2').split(','))
dev = torch.device('cpu'); torch.set_num_threads(1)
df = pd.read_csv(P_tr, sep=';'); gmd, gmt, gm, gs, gmax, gmin = compute_global_stats(df)
kw = dict(max_dose=gmd, max_time=gmt, max_value=gmax, min_value=gmin, global_mean=gm, global_std=gs)
dose_t = np.unique(np.concatenate([pd.read_csv(f, sep=';').query('EVID > 0').TIME.values for f in [P_tr] + tests])) / gmt
n_grid = int(round(gmt / 0.25)) + 1
t_dense = torch.tensor(np.unique(np.concatenate([np.linspace(0, 1, n_grid), dose_t]).astype(np.float32)), dtype=torch.float32)


def build(h_enc, m_enc, layers, add, learn_cov):
    enc = Encoder_Transformer(dim_latent=LAT, dim_parameters_IC=IC, dim_parameters_dynamic=DYN, dim_cov=3, model_dim=m_enc, hidden_dim=h_enc,
                              num_heads=2, num_layers=layers, dropout=0.1 if layers == 1 else 0.05, cov_diag_epsilon=1e-5,
                              learn_prior_mean=True, learn_prior_covariance=learn_cov, diagonal_only=False, IC_dose_dependent=False)
    f = ODEFunc(dim_latent=LAT, dim_parameters_IC=IC, dim_parameters_dynamic=DYN, hid_dim=32, drug_dim=1)
    return {'func': f, 'encoder': enc, 'reducer': SimpleDecoder(LAT), 'noise': TrainableNoise(init_add_std=add, init_prop_std=0.01)}


S = torch.load(ckpt, map_location='cpu', weights_only=False)
med, eb = build(8, 16, 1, 0.2, False), build(16, 32, 2, 0.1, True)
for M, key in ((med, 'med'), (eb, 'eb')):
    for k, m in M.items():
        m.load_state_dict(S[key][k]); m.eval()

dte = TrajectoryDataset(P_ev, **kw); raw = pd.read_csv(P_ev, sep=';')
obs = raw[raw.EVID == 0]
info = obs.groupby('ID').agg(form=('FORM', 'first'))
if kind == 'part2':
    auc = obs.groupby(['ID', 'VISIT']).AUC.first(); t2 = obs.query('VISIT == 2').groupby('ID').TIME.min()
    res = dict(id=[], true_auc_v1=[], pred_auc_v1=[], true_auc_v2=[], pred_auc_v2=[])
else:
    auc = obs.groupby('ID').AUC.first()
    res = dict(id=[], true_auc=[], pred_auc=[])
with torch.no_grad():
    for batch in DataLoader(dte, batch_size=50, shuffle=False, collate_fn=make_collate_fn(dte.global_max_len)):
        ids, tr, t_pad, x_pad, t_enc, x_enc, t_cut, x_cut, m_enc, cov, dose, dts, evid = preprocess_batch(batch, dev, truncation=1)
        x_in = normalize_encoder_input(x_pad, t_pad, x_pad, cov, dose, m_enc, dts, evid, med['encoder'], med['func'], med['reducer'], t_dense, gm, gs)
        k, z0, mu_q, L_q, mu_p, L_p, mL1, rf = encode_latent(eb['encoder'], t_pad, x_in, cov, dose, m_enc, enable_vae=False, enable_ae=True, enable_onlymedian=False)
        f = prepare_ode_input(x_pad, k, eb['func'], cov, dose, dts, evid, enable_ae=True, enable_onlymedian=False)
        idl = ids.tolist()
        if kind == 'part2':
            grids = []
            for i in idl:
                tau = 12.0 if info.loc[i].form == 1 else 24.0; g = np.round(np.arange(0, tau + 1e-9, 0.05), 4)
                grids.append(np.concatenate([g, t2.loc[i] + g]) / gmt)
            L = max(len(g) for g in grids); tg = torch.tensor(np.stack([np.pad(g, (0, L - len(g)), mode='edge') for g in grids]), dtype=torch.float32)
            pred = make_predictions(tg, t_dense, k, f, eb['reducer'], gm, gs)[0].numpy()
            for j, i in enumerate(idl):
                tau = 12.0 if info.loc[i].form == 1 else 24.0; g = np.round(np.arange(0, tau + 1e-9, 0.05), 4); n = len(g)
                res['id'].append(int(i)); res['true_auc_v1'].append(float(auc.loc[(i, 1)])); res['true_auc_v2'].append(float(auc.loc[(i, 2)]))
                res['pred_auc_v1'].append(float(np.trapezoid(pred[j, :n], g))); res['pred_auc_v2'].append(float(np.trapezoid(pred[j, n:2 * n], g)))
        else:
            g24 = np.round(np.arange(0, 24 + 1e-9, 0.05), 4)
            tg = torch.tensor(np.tile(g24 / gmt, (len(idl), 1)), dtype=torch.float32)
            pred = make_predictions(tg, t_dense, k, f, eb['reducer'], gm, gs)[0].numpy()
            for j, i in enumerate(idl):
                tau = 12.0 if info.loc[i].form == 1 else 24.0; m = g24 <= tau + 1e-9
                res['id'].append(int(i)); res['true_auc'].append(float(auc.loc[i])); res['pred_auc'].append(float(np.trapezoid(pred[j, m], g24[m])))
summ = {}
for v in (('v1', 'v2') if kind == 'part2' else ('',)):
    t, p = (np.array(res[f'{a}_auc{"_" + v if v else ""}']) for a in ('true', 'pred'))
    e = p / t - 1
    summ[v or 'factual'] = dict(median_abs=100 * float(np.median(np.abs(e))), rmspe=100 * float(np.sqrt(np.mean(e ** 2))),
                                bias=100 * float(np.median(e)), n_gt100=int((np.abs(e) > 1).sum()), n=len(e))
json.dump(dict(kind=kind, ckpt=ckpt, train=P_tr, eval=P_ev, dims=[IC, DYN, LAT], per_patient=res, summary=summ), open(out, 'w'))
print(f'{os.path.basename(ckpt)} on {os.path.basename(P_ev)}: ' + '; '.join(f'{k} median |AUC err| {v["median_abs"]:.1f} % RMSPE {v["rmspe"]:.1f} % (n {v["n"]})' for k, v in summ.items()), flush=True)
