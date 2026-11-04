"""EB-VAE (Baaz et al. 2026, public code) in the iScience FORMAT (PharmaNODE 2026-10-06): the encoder sees only
the FIRST dosing interval of a regimen and the model predicts the rest of the SAME regimen. Same data stream as run_regimen.py
(training patients / noise identical); the encoder restriction uses our opt-in patch P3 (BAAZ_ENC_COL=ENC: the encoder sees
exactly the flagged observations, the loss all of them -- ours/PATCHES.md), flag = observation before the second dose.
Test: the 400 test patients observed under EVERY regimen (noise stream seed + 4242, identical to scripts/regimen/*_regimen*.py
--protocol firstint in PharmaNODE), encoded from that regimen's first interval (0-8 h; 0-3 h for time400/800), posterior mean,
predicted under the same regimen; median |AUC(t2-30 h) err| and nRMSE on [t2, 30 h] vs the noise-free curve.
    run_regimen_firstint.py <n_per_arm> <dense|sparse> <seed> <out.json> [E1 E2]"""
import os, sys, json, time, random
os.environ['BAAZ_ENC_COL'] = 'ENC'
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'ours/regimen')
import gen_regimens as G
from torch.utils.data import DataLoader
from lib.utils.utils_preprocess import TrajectoryDataset, make_collate_fn, compute_global_stats, prepare_optimizer
from lib.utils.utils_training import train_loop_model
from lib.utils.utils_shared import preprocess_batch, encode_latent, prepare_ode_input, make_predictions, normalize_encoder_input
from lib.models.NNmodels import Encoder_Transformer, ODEFunc, SimpleDecoder, TrainableNoise
NA, DESIGN, seed, OUT = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
E1, E2 = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (100, 300)
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); dev = torch.device('cpu')
SAMP = np.arange(0, 30.25, 0.5) if DESIGN == 'dense' else np.array([0, 2, 7.5, 15.5, 23.5, 28.0])
WD = os.path.splitext(OUT)[0] + '_data'; os.makedirs(WD, exist_ok=True); t0 = time.time()
def t2_of(times):
    return float(sorted(times)[1])
def make_csv(path, arm, times, amounts, noisy_obs, ids, cut):
    rows = []
    for j, i in enumerate(ids):
        for t, a in zip(times, amounts[j]):
            rows.append(dict(ID=int(i), TIME=float(t), DV=np.nan, AMT=float(a), EVID=1, COV_IND=0, TREATMENT=1 + int(arm[j] == 800), ENC=0))
        for t, y in zip(SAMP, noisy_obs[j]):
            rows.append(dict(ID=int(i), TIME=float(t), DV=float(y), AMT=0.0, EVID=0, COV_IND=0, TREATMENT=1 + int(arm[j] == 800), ENC=int(t < cut)))
    pd.DataFrame(rows).sort_values(['ID', 'TIME', 'EVID'], ascending=[True, True, False]).to_csv(path, sep=';', index=False)
def sample_obs(P, times, amounts, rng):
    t_eval, c = G.simulate(P, times, amounts)
    return t_eval, c, G.noisy(np.stack([np.interp(SAMP, t_eval, ci) for ci in c]), rng)
rng = np.random.default_rng(seed)
# ---- training data: run_regimen.py's exact sequence; the encoder flag = first dosing interval (t < 8 h) ----
n_tr = 2 * NA; P_tr = G.draw(n_tr, rng); arm_tr = np.where(np.arange(n_tr) % 2 == 0, 400.0, 800.0)
A_tr = np.repeat(arm_tr[:, None], 4, 1); _, _, y_tr = sample_obs(P_tr, [0, 8, 16, 24], A_tr, rng)
make_csv(f'{WD}/train.csv', arm_tr, [0, 8, 16, 24], A_tr, y_tr, np.arange(1, n_tr + 1), cut=t2_of([0, 8, 16, 24]))
# ---- test patients (same draws), observed under every regimen from the separate stream ----
n_te = 400; P_te = G.draw(n_te, rng); arm_te = np.where(np.arange(n_te) % 2 == 0, 400.0, 800.0); ids_te = np.arange(10001, 10001 + n_te)
A_te = np.repeat(arm_te[:, None], 4, 1); _, _, y_te = sample_obs(P_te, [0, 8, 16, 24], A_te, rng)
truth, regs = {}, {}
for name, (times, amts) in G.REGIMENS.items():
    A = A_te if amts is None else np.tile(np.array(amts, float), (n_te, 1)); regs[name] = (times, A)
    t_eval, truth[name] = G.simulate(P_te, times, A)
rng_r = np.random.default_rng(seed + 4242)
y_obs = {name: G.noisy(np.stack([np.interp(SAMP, t_eval, c) for c in truth[name]]), rng_r) for name in G.REGIMENS}
for name, (times, A) in regs.items():
    make_csv(f'{WD}/test_obs_{name}.csv', arm_te, times, A, y_obs[name], ids_te, cut=t2_of(times))
# ---- their pipeline (as run_regimen.py) ----
df = pd.read_csv(f'{WD}/train.csv', sep=';'); gmd, gmt, gm, gs, gmax, gmin = compute_global_stats(df)
kw = dict(max_dose=gmd, max_time=gmt, max_value=gmax, min_value=gmin, global_mean=gm, global_std=gs)
ds = TrajectoryDataset(f'{WD}/train.csv', **kw)
tl = DataLoader(ds, batch_size=max(1, int(len(ds) * 0.05)), shuffle=True, collate_fn=make_collate_fn(ds.global_max_len))
all_dose_t = np.unique(np.concatenate([np.array(r[0], float) for r in G.REGIMENS.values()])) / gmt
t_dense = torch.tensor(np.unique(np.concatenate([np.linspace(0, 1, 121), all_dose_t]).astype(np.float32)), dtype=torch.float32)
def build(h_enc, m_enc, layers, add, learn_cov):
    enc = Encoder_Transformer(dim_latent=2, dim_parameters_IC=1, dim_parameters_dynamic=2, dim_cov=1, model_dim=m_enc, hidden_dim=h_enc,
        num_heads=2, num_layers=layers, dropout=0.1 if layers == 1 else 0.05, cov_diag_epsilon=1e-5, learn_prior_mean=True,
        learn_prior_covariance=learn_cov, diagonal_only=False, IC_dose_dependent=False)
    return {'func': ODEFunc(dim_latent=2, dim_parameters_IC=1, dim_parameters_dynamic=2, hid_dim=32, drug_dim=1), 'encoder': enc,
            'reducer': SimpleDecoder(2), 'noise': TrainableNoise(init_add_std=add, init_prop_std=0.01)}
med = build(8, 16, 1, 0.2, False)
o1, s1, p1 = prepare_optimizer(med, dev, lr=1e-3, factor=0.8, patience=20, min_lr=1e-5, prior_lr_factor=1.0)
train_loop_model(med, med, dataloader=tl, dataset_train=df, main_params=p1, optimizer=o1, scheduler=s1, t_dense=t_dense, n_epochs=E1,
                 warmup_epochs_noise=1000, warmup_epochs_iiv=0, smoothing_start_epoch=0, enable_ae=False, enable_vae=False,
                 enable_onlymedian=True, normalization=False, truncation=1, print_epoch=10)
eb = build(16, 32, 2, 0.1, True)
o2, s2, p2 = prepare_optimizer(eb, dev, lr=1e-3, factor=0.8, patience=10, min_lr=1e-5, prior_lr_factor=5.0)
train_loop_model(models_norm=med, models_eval=eb, dataloader=tl, dataset_train=df, main_params=p2, optimizer=o2, scheduler=s2, t_dense=t_dense,
                 n_epochs=E2, warmup_epochs_noise=200, warmup_epochs_iiv=0, smoothing_start_epoch=0, enable_ae=False, enable_vae=True,
                 enable_onlymedian=False, normalization=True, truncation=1, print_epoch=10)
minutes_train = (time.time() - t0) / 60
for M in (med, eb):
    for m in M.values(): m.eval()
EG = np.arange(0, 30.0001, 0.25); tg_norm = torch.tensor(EG / gmt, dtype=torch.float32)
def loader(path):
    d = TrajectoryDataset(path, **kw); return DataLoader(d, batch_size=100, shuffle=False, collate_fn=make_collate_fn(d.global_max_len))
res = dict(model='ebvae', protocol='firstint', N_per_arm=NA, design=DESIGN, seed=seed, n_samples_per_patient=len(SAMP),
           minutes_train=minutes_train, individual={}, window_start={})
with torch.no_grad():
    for name in G.REGIMENS:
        times, A = regs[name]; cut = t2_of(times); preds = []
        for b in loader(f'{WD}/test_obs_{name}.csv'):
            ids, tr, t_pad, x_pad, te_, xe_, tc, xc, m_enc, cov, dose, dts, evid = preprocess_batch(b, dev, truncation=1)
            x_in = normalize_encoder_input(x_pad, t_pad, x_pad, cov, dose, m_enc, dts, evid, med['encoder'], med['func'], med['reducer'], t_dense, gm, gs)
            k, *_ = encode_latent(eb['encoder'], t_pad, x_in, cov, dose, m_enc, enable_vae=False, enable_ae=True, enable_onlymedian=False)
            f = prepare_ode_input(x_pad, k, eb['func'], cov, dose, dts, evid, enable_ae=True, enable_onlymedian=False)
            p, _ = make_predictions(tg_norm.unsqueeze(0).expand(len(ids), -1), t_dense, k, f, eb['reducer'], gm, gs); preds.append(p.numpy())
        w = EG >= cut - 1e-9
        pred = np.concatenate(preds)[:, w]; tru = np.stack([np.interp(EG, t_eval, c) for c in truth[name]])[:, w]
        auc_p, auc_t = np.trapezoid(pred, EG[w], axis=1), np.trapezoid(tru, EG[w], axis=1)
        r = auc_p / auc_t - 1; nrmse = np.sqrt(np.mean((pred - tru) ** 2, 1)) / np.mean(tru, 1)
        res['individual'][name] = dict(auc_abs_median=100 * float(np.median(np.abs(r))), nrmse_median=100 * float(np.median(nrmse)),
                                       auc_bias_median=100 * float(np.median(r)), auc_err_per_patient=[float(x) for x in r],
                                       auc_pred_per_patient=[float(x) for x in auc_p])
        res['window_start'][name] = cut
        print(f'{name:8s} first {cut:g} h observed -> |AUC({cut:g}-30 h) err| {res["individual"][name]["auc_abs_median"]:5.1f}%  '
              f'nRMSE {res["individual"][name]["nrmse_median"]:5.1f}%  bias {res["individual"][name]["auc_bias_median"]:+5.1f}%', flush=True)
json.dump(res, open(OUT, 'w'), indent=1); print(f'REGIMEN_DONE {(time.time() - t0) / 60:.1f} min')
