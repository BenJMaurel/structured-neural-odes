"""TITRATED COPY of run_regimen.py (PharmaNODE 2026-10-05): identical except (1) argument <rho>: 'none' = the
original alternating arms (control, must reproduce run_regimen.py), a number = gen_regimens.titrate arms (fast eliminators get
800 mg; own random stream, so patients and noise draws are unchanged; test arms stay alternating); (2) per-patient AUC errors
and predicted AUCs are stored (dose elasticity). Original docstring:
Does the EB-VAE (Baaz et al. 2026, public code, their simulated-study pipeline, NO patches) generalise to unseen
regimens, and with how much data?  Their own design: linear 1-cpt PK with absorption, correlated IIV (ka, ke, C0),
training arms 400 / 800 mg x 4 doses at 0, 8, 16, 24 h, recording 0-30 h; their two-stage training (median model 100 ep,
EB-VAE 300 ep, normalisation on), truncation 1 (the encoder sees all of the patient's observations).
Factors: N per arm; sampling 'dense' (theirs: every 0.5 h, 61 samples) or 'sparse' (6 samples: 0, 2, 7.5, 15.5, 23.5, 28 h).
Test: 400 NEW patients, encoded from their own training-regimen data (same sampling design), predicted under regimens
  train (factual) | amt350 / amt600 / amt850 (new amounts, same times) | time400 / time800 (doses at 0, 3, 8 h, as their Fig 3)
  INDIVIDUAL: posterior-mean latent, ODE under the new regimen's doses; vs the same patient's noise-free true curve:
              |AUC(0-30) error| and curve nRMSE (RMSE / mean true conc), medians over patients.
  POPULATION: 2000 draws from the learned prior (their VPC mode, sample_posterior=False) vs the true noise-free population
              (2000 new patients): mean |error| of the 10/50/90 % quantile curves / true median curve mean.
    run_regimen_titrated.py <n_per_arm> <dense|sparse> <seed> <rho|none> <out.json> [E1 E2]"""
import os, sys, json, time, random
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd()); sys.path.insert(0, 'ours/regimen')
import gen_regimens as G
from torch.utils.data import DataLoader
from lib.utils.utils_preprocess import TrajectoryDataset, make_collate_fn, compute_global_stats, prepare_optimizer
from lib.utils.utils_training import train_loop_model
from lib.utils.utils_shared import preprocess_batch, encode_latent, prepare_ode_input, make_predictions, normalize_encoder_input
from lib.models.NNmodels import Encoder_Transformer, ODEFunc, SimpleDecoder, TrainableNoise
PRIOR_NO_DOSE = bool(os.environ.get('BAAZ_PRIOR_NO_DOSE'))
if PRIOR_NO_DOSE:   # PharmaNODE (prior test on the design of Baaz et al.; as run_regimen_tacro.py, 0.00zz58): the prior
    _get_prior = Encoder_Transformer.get_prior        # mean of the parameters ignores the first amount (it still reads the covariates), in
    def _get_prior_no_dose(self, cov, dose_tensor, learn_prior_mean, batch_size=None):   # both training stages (the only caller passes the dose)
        return _get_prior(self, cov, torch.zeros_like(dose_tensor), learn_prior_mean, batch_size=batch_size)
    Encoder_Transformer.get_prior = _get_prior_no_dose
ENC_NORM_DOSE = os.environ.get('BAAZ_ENC_NORM') == 'dose'
if ENC_NORM_DOSE:   # PharmaNODE (normalisation test): the stage-2 encoder input is the concentrations divided by the patient's
    import lib.utils.utils_shared as _US, lib.utils.utils_training as _UT          # first amount (relative to the largest) and by the global
    from lib.utils.utils_preprocess import destandardize_concentration as _destd    # mean concentration, as the dose-normalised encoder of the
    def _norm_by_dose(x_encoder, t_encoder, x_padded, cov, dose_tensor, mask, dose_times_list, evid,     # dose-event model, INSTEAD of the
                      encoder_med, func_med, reducer_med, t_dense, global_mean, global_std):          # concentrations divided by the stage-1
        conc = _destd(x_encoder, global_mean, global_std)                                             # population prediction under the
        return (conc / dose_tensor[:, 0:1].clamp(min=1e-6) / global_mean).detach()                    # patient's own regimen; all else theirs
    _US.normalize_encoder_input = _norm_by_dose; _UT.normalize_encoder_input = _norm_by_dose; normalize_encoder_input = _norm_by_dose
NA, DESIGN, seed, OUT = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[5]
RHO = None if sys.argv[4] == 'none' else float(sys.argv[4])
E1, E2 = (int(sys.argv[6]), int(sys.argv[7])) if len(sys.argv) > 7 else (100, 300)
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); dev = torch.device('cpu')
SAMP = np.arange(0, 30.25, 0.5) if DESIGN == 'dense' else np.array([0, 2, 7.5, 15.5, 23.5, 28.0])
WD = os.path.splitext(OUT)[0] + '_data'; os.makedirs(WD, exist_ok=True); t0 = time.time()
def make_csv(path, P, arm, times, amounts, noisy_obs, ids):
    rows = []
    for j, i in enumerate(ids):
        for t, a in zip(times, amounts[j]):
            rows.append(dict(ID=int(i), TIME=float(t), DV=np.nan, AMT=float(a), EVID=1, COV_IND=0, TREATMENT=1 + int(arm[j] == 800)))
        for t, y in zip(SAMP, noisy_obs[j]):
            rows.append(dict(ID=int(i), TIME=float(t), DV=float(y), AMT=0.0, EVID=0, COV_IND=0, TREATMENT=1 + int(arm[j] == 800)))
    pd.DataFrame(rows).sort_values(['ID', 'TIME', 'EVID'], ascending=[True, True, False]).to_csv(path, sep=';', index=False)
def sample_obs(P, times, amounts, rng):
    t_eval, c = G.simulate(P, times, amounts)
    return t_eval, c, G.noisy(np.stack([np.interp(SAMP, t_eval, ci) for ci in c]), rng)
rng = np.random.default_rng(seed)
# ---- training data (arms alternate 400 / 800) ----
n_tr = 2 * NA; P_tr = G.draw(n_tr, rng)
if RHO is None:
    arm_tr, titr_corr = np.where(np.arange(n_tr) % 2 == 0, 400.0, 800.0), None
else:
    arm_tr, titr_corr = G.titrate(P_tr, RHO, seed); print(f'TITRATED training arms: rho {RHO}, corr(log ke, 800 mg) {titr_corr:+.2f}', flush=True)
A_tr = np.repeat(arm_tr[:, None], 4, 1); _, _, y_tr = sample_obs(P_tr, [0, 8, 16, 24], A_tr, rng)
make_csv(f'{WD}/train.csv', P_tr, arm_tr, [0, 8, 16, 24], A_tr, y_tr, np.arange(1, n_tr + 1))
# ---- test patients: factual data (same design) + counterfactual truth under every regimen ----
n_te = 400; P_te = G.draw(n_te, rng); arm_te = np.where(np.arange(n_te) % 2 == 0, 400.0, 800.0); ids_te = np.arange(10001, 10001 + n_te)
A_te = np.repeat(arm_te[:, None], 4, 1); _, _, y_te = sample_obs(P_te, [0, 8, 16, 24], A_te, rng)
make_csv(f'{WD}/test_factual.csv', P_te, arm_te, [0, 8, 16, 24], A_te, y_te, ids_te)
truth, regs = {}, {}
for name, (times, amts) in G.REGIMENS.items():
    A = A_te if amts is None else np.tile(np.array(amts, float), (n_te, 1)); regs[name] = (times, A)
    t_eval, truth[name] = G.simulate(P_te, times, A)
    make_csv(f'{WD}/test_{name}.csv', P_te, arm_te, times, A, y_te, ids_te)     # same observations, regimen's doses
# ---- their pipeline ----
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
res = dict(N_per_arm=NA, design=DESIGN, seed=seed, n_samples_per_patient=len(SAMP), minutes_train=minutes_train, titrate=RHO, titrate_corr=titr_corr, individual={}, population={}, **({'prior_no_dose': True} if PRIOR_NO_DOSE else {}), **({'enc_norm': 'dose'} if ENC_NORM_DOSE else {}))
curves = {}   # opt-in (DUMP_CURVES=1): predicted test curves on EG per regimen, for patient-level figures
with torch.no_grad():
    for name in G.REGIMENS:
        preds = []
        for bf, br in zip(loader(f'{WD}/test_factual.csv'), loader(f'{WD}/test_{name}.csv')):
            ids, tr, t_pad, x_pad, te_, xe_, tc, xc, m_enc, cov, dose, dts, evid = preprocess_batch(bf, dev, truncation=1)
            x_in = normalize_encoder_input(x_pad, t_pad, x_pad, cov, dose, m_enc, dts, evid, med['encoder'], med['func'], med['reducer'], t_dense, gm, gs)
            k, *_ = encode_latent(eb['encoder'], t_pad, x_in, cov, dose, m_enc, enable_vae=False, enable_ae=True, enable_onlymedian=False)
            _, _, t_pad_r, x_pad_r, _, _, _, _, _, cov_r, dose_r, dts_r, evid_r = preprocess_batch(br, dev, truncation=1)
            f = prepare_ode_input(x_pad_r, k, eb['func'], cov_r, dose_r, dts_r, evid_r, enable_ae=True, enable_onlymedian=False)
            p, _ = make_predictions(tg_norm.unsqueeze(0).expand(len(ids), -1), t_dense, k, f, eb['reducer'], gm, gs); preds.append(p.numpy())
        pred = np.concatenate(preds); tru = np.stack([np.interp(EG, t_eval, c) for c in truth[name]]); curves[name] = pred
        auc_e = np.abs(np.trapezoid(pred, EG, axis=1) / np.trapezoid(tru, EG, axis=1) - 1)
        nrmse = np.sqrt(np.mean((pred - tru) ** 2, 1)) / np.mean(tru, 1)
        res['individual'][name] = dict(auc_abs_median=100 * float(np.median(auc_e)), nrmse_median=100 * float(np.median(nrmse)),
                                       auc_bias_median=100 * float(np.median(np.trapezoid(pred, EG, axis=1) / np.trapezoid(tru, EG, axis=1) - 1)),
                                       auc_err_per_patient=[float(x) for x in np.trapezoid(pred, EG, axis=1) / np.trapezoid(tru, EG, axis=1) - 1],
                                       auc_pred_per_patient=[float(x) for x in np.trapezoid(pred, EG, axis=1)])
        # population: prior draws (their VPC mode) vs a fresh true population under this regimen
        times, A = regs[name]; n_pop = 2000; P_pop = G.draw(n_pop, np.random.default_rng(seed + 777))
        A_pop = np.tile(A[:1], (n_pop, 1)) if name != 'train' else np.repeat(np.where(np.arange(n_pop) % 2 == 0, 400.0, 800.0)[:, None], len(times), 1)
        _, c_pop = G.simulate(P_pop, times, A_pop); q_true = np.percentile(np.stack([np.interp(EG, t_eval, c) for c in c_pop]), [10, 50, 90], 0)
        sims = []
        for _ in range(n_pop // 400):
            for br in loader(f'{WD}/test_{name}.csv'):
                ids, tr, t_pad, x_pad, te_, xe_, tc, xc, m_enc, cov, dose, dts, evid = preprocess_batch(br, dev, truncation=1)
                if name == 'train':
                    pass
                k, *_ = encode_latent(eb['encoder'], t_pad, x_pad, cov, dose, m_enc, enable_vae=True, enable_ae=False, enable_onlymedian=False, sample_posterior=False)
                f = prepare_ode_input(x_pad, k, eb['func'], cov, dose, dts, evid, enable_ae=False, enable_onlymedian=False)
                p, _ = make_predictions(tg_norm.unsqueeze(0).expand(len(ids), -1), t_dense, k, f, eb['reducer'], gm, gs); sims.append(p.numpy())
        q_pred = np.percentile(np.concatenate(sims), [10, 50, 90], 0)
        res['population'][name] = {f'q{q}_mae_rel': 100 * float(np.mean(np.abs(q_pred[j] - q_true[j])) / np.mean(q_true[1])) for j, q in enumerate((10, 50, 90))}
        print(f'{name:8s} individual |AUC err| {res["individual"][name]["auc_abs_median"]:5.1f}%  nRMSE {res["individual"][name]["nrmse_median"]:5.1f}%  '
              f'| population quantile error q10/q50/q90 ' + ' / '.join(f'{v:.1f}%' for v in res['population'][name].values()), flush=True)
if os.environ.get('DUMP_CURVES'):
    np.savez_compressed(os.path.splitext(OUT)[0] + '_curves.npz', EG=EG, **curves)
if os.environ.get('BAAZ_DUMP_LATENT'):   # PharmaNODE (latent test), opt-in, after every evaluation above: each test patient
    rng_cf = np.random.default_rng(seed + 9091)          # observed under the training schedule at 400 mg, again at 400 mg (fresh noise:
    obs, enc = {}, {}                                    # the noise floor) and at 800 mg (own random stream; seed + 777 is the population
    for tag, a_mg in (('400a', 400.0), ('400b', 400.0), ('800', 800.0)):    # draws), encoded exactly as in the evaluation above (median-
        arm_cf = np.full(n_te, a_mg); A_cf = np.repeat(arm_cf[:, None], 4, 1)                  # model normalisation, AE-mode posterior mean)
        _, _, y_cf = sample_obs(P_te, [0, 8, 16, 24], A_cf, rng_cf)
        make_csv(f'{WD}/test_cf{tag}.csv', P_te, arm_cf, [0, 8, 16, 24], A_cf, y_cf, ids_te); obs[tag] = y_cf
    with torch.no_grad():
        for tag in obs:
            ks = []
            for bf in loader(f'{WD}/test_cf{tag}.csv'):
                ids, tr, t_pad, x_pad, te_, xe_, tc, xc, m_enc, cov, dose, dts, evid = preprocess_batch(bf, dev, truncation=1)
                x_in = normalize_encoder_input(x_pad, t_pad, x_pad, cov, dose, m_enc, dts, evid, med['encoder'], med['func'], med['reducer'], t_dense, gm, gs)
                k, *_ = encode_latent(eb['encoder'], t_pad, x_in, cov, dose, m_enc, enable_vae=False, enable_ae=True, enable_onlymedian=False)
                ks.append(k.detach().numpy())
            enc[tag] = np.concatenate(ks)
    np.savez_compressed(os.path.splitext(OUT)[0] + '_latent.npz', P_te=P_te, samp=SAMP, ids=ids_te,
                        **{f'y_{t}': v for t, v in obs.items()}, **{f'k_{t}': v for t, v in enc.items()})
    print(f'LATENT_DUMPED {os.path.splitext(OUT)[0]}_latent.npz', flush=True)
json.dump(res, open(OUT, 'w'), indent=1); print(f'REGIMEN_DONE {(time.time() - t0) / 60:.1f} min')
