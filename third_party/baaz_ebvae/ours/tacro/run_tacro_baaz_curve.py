"""CONVERGENCE COPY of run_tacro_baaz.py (PharmaNODE 2026-10-07). Same data, seeds, stage 1 and stage-2 settings; stage 2 is trained in CHUNKS ending at the epochs listed in
<E2 checkpoints> and the test set is scored after each chunk. Chunking is exactly an uninterrupted run: the optimizer, the
plateau scheduler and the global RNG stream carry over (the RNG state is saved and restored around every evaluation, since
creating a DataLoader iterator draws from it), and the noise warm-up counts total epochs (noise trainable from stage-2
epoch 200, as in one 300+ epoch call). Hence the epoch-300 evaluation must equal the original run's eval JSON.
    run_tacro_baaz_curve.py <seed> <train.csv> <out_dir> <E1> <E2 checkpoints, e.g. 300,400,...,1000> <test.csv> [...]
Original docstring:
EB-VAE (Baaz et al. 2026, public code + opt-in patches P1-P3, ours/PATCHES.md) on the PharmaNODE switch-design
benchmark. Their MULTI-DOSE pipeline (scripts/main_simulated_data.py): stage 1 'median' model (k = prior mean; small
encoder 16/8/2 heads/1 layer, ODE hidden 32, add/prop init 0.2/0.01, lr 1e-3, patience 20, prior_lr_factor 1;
warm-up noise 1000 -> noise frozen), stage 2 EB-VAE (encoder 32/16/2 heads/2 layers, ODE hidden 32, learned prior mean +
full covariance, add/prop init 0.1/0.01, lr 1e-3, patience 10, prior_lr_factor 5, warm-up noise 200, input-response
normalisation ON), batch_fraction 0.05. Epochs E1 (their commented default 100) and E2 (their default 300).
Adaptations (documented): covariates FORM, CYP, HT/35 (P2); dose change d1 -> d2 within a patient (P1); encoder sees
exactly the visit-1 samples at 0, 1, 3 h (P3); dose-pulse width initialised to their physical 1.5 h; grid step 0.25 h.
Prediction (their compute_test_metrics logic, repaired): encoder on the flagged samples (normalised as in training,
median-model response), posterior mean (AE mode), ODE through the patient's own dosing timeline; AUC = trapezoid of the
predicted curve over [0, tau] (visit 1) and [t2, t2 + tau] (visit 2), 0.05-h grid. Raw (non-EMA) weights, as their metric.
    run_tacro_baaz.py <seed> <train.csv> <out_dir> <E1> <E2> <test_in.csv> [<test_lo.csv> <test_hi.csv>]"""
import os, sys, json, time, random
os.environ.update(BAAZ_PER_DOSE='1', BAAZ_COV_COLS='FORM,CYP,HTN', BAAZ_ENC_COL='ENC')
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
from torch.utils.data import DataLoader
from lib.utils.utils_preprocess import TrajectoryDataset, make_collate_fn, compute_global_stats, prepare_optimizer
from lib.utils.utils_training import train_loop_model
from lib.utils.utils_shared import preprocess_batch, encode_latent, prepare_ode_input, make_predictions, normalize_encoder_input
from lib.models.NNmodels import Encoder_Transformer, ODEFunc, SimpleDecoder, TrainableNoise
seed, P_tr, out, E1 = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]); CKPTS = [int(x) for x in sys.argv[5].split(',')]; tests = sys.argv[6:]
assert CKPTS == sorted(CKPTS) and CKPTS[0] > 0
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); os.makedirs(out, exist_ok=True); dev = torch.device('cpu')
df = pd.read_csv(P_tr, sep=';'); gmd, gmt, gm, gs, gmax, gmin = compute_global_stats(df)
kw = dict(max_dose=gmd, max_time=gmt, max_value=gmax, min_value=gmin, global_mean=gm, global_std=gs)
ds = TrajectoryDataset(P_tr, **kw)
tl = DataLoader(ds, batch_size=max(1, int(len(ds) * 0.05)), shuffle=True, collate_fn=make_collate_fn(ds.global_max_len))
dose_t = np.unique(np.concatenate([pd.read_csv(f, sep=';').query('EVID > 0').TIME.values for f in [P_tr] + tests])) / gmt
n_grid = int(round(gmt / 0.25)) + 1                                     # 0.25-h step, as their 30-h / 120-point design
t_dense = torch.tensor(np.unique(np.concatenate([np.linspace(0, 1, n_grid), dose_t]).astype(np.float32)), dtype=torch.float32)   # unique AFTER the float32 cast (strictly increasing)
SIG0 = float(np.log(0.05 * 30.0 / gmt))                                 # their pulse width 0.05 x 30 h = 1.5 h, physical
def build(h_enc, m_enc, layers, add, learn_cov):
    enc = Encoder_Transformer(dim_latent=2, dim_parameters_IC=1, dim_parameters_dynamic=2, dim_cov=3, model_dim=m_enc, hidden_dim=h_enc,
                              num_heads=2, num_layers=layers, dropout=0.1 if layers == 1 else 0.05, cov_diag_epsilon=1e-5,
                              learn_prior_mean=True, learn_prior_covariance=learn_cov, diagonal_only=False, IC_dose_dependent=False)
    f = ODEFunc(dim_latent=2, dim_parameters_IC=1, dim_parameters_dynamic=2, hid_dim=32, drug_dim=1)
    with torch.no_grad(): f.log_sigma.fill_(SIG0)
    return {'func': f, 'encoder': enc, 'reducer': SimpleDecoder(2), 'noise': TrainableNoise(init_add_std=add, init_prop_std=0.01)}
t0 = time.time(); print(f'{len(ds)} training patients, max time {gmt:.0f} h, grid {len(t_dense)} points, pulse width init {np.exp(SIG0) * gmt:.2f} h', flush=True)
med = build(8, 16, 1, 0.2, False)
o1, s1, p1 = prepare_optimizer(med, dev, lr=1e-3, factor=0.8, patience=20, min_lr=1e-5, prior_lr_factor=1.0)
train_loop_model(med, med, dataloader=tl, dataset_train=df, main_params=p1, optimizer=o1, scheduler=s1, t_dense=t_dense, n_epochs=E1,
                 warmup_epochs_noise=1000, warmup_epochs_iiv=0, smoothing_start_epoch=0, enable_ae=False, enable_vae=False,
                 enable_onlymedian=True, normalization=False, truncation=1, print_epoch=10)
print(f'stage 1 done {(time.time() - t0) / 60:.1f} min', flush=True)
eb = build(16, 32, 2, 0.1, True)
o2, s2, p2 = prepare_optimizer(eb, dev, lr=1e-3, factor=0.8, patience=10, min_lr=1e-5, prior_lr_factor=5.0)
def evaluate(E):
    rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
    for M in (med, eb):
        for m in M.values(): m.eval()
    for P_te in tests:
        dte = TrajectoryDataset(P_te, **kw); raw = pd.read_csv(P_te, sep=';')
        info = raw[raw.EVID == 0].groupby('ID').agg(form=('FORM', 'first')); auc = raw[raw.EVID == 0].groupby(['ID', 'VISIT']).AUC.first()
        t2 = raw[raw.EVID == 0].query('VISIT == 2').groupby('ID').TIME.min()
        res = dict(id=[], true_auc_v1=[], pred_auc_v1=[], true_auc_v2=[], pred_auc_v2=[])
        with torch.no_grad():
            for batch in DataLoader(dte, batch_size=50, shuffle=False, collate_fn=make_collate_fn(dte.global_max_len)):
                ids, tr, t_pad, x_pad, t_enc, x_enc, t_cut, x_cut, m_enc, cov, dose, dts, evid = preprocess_batch(batch, dev, truncation=1)
                x_in = normalize_encoder_input(x_pad, t_pad, x_pad, cov, dose, m_enc, dts, evid, med['encoder'], med['func'], med['reducer'], t_dense, gm, gs)
                k, z0, mu_q, L_q, mu_p, L_p, mL1, rf = encode_latent(eb['encoder'], t_pad, x_in, cov, dose, m_enc, enable_vae=False, enable_ae=True, enable_onlymedian=False)
                f = prepare_ode_input(x_pad, k, eb['func'], cov, dose, dts, evid, enable_ae=True, enable_onlymedian=False)
                idl = ids.tolist(); grids = []
                for i in idl:
                    tau = 12.0 if info.loc[i].form == 1 else 24.0; g = np.round(np.arange(0, tau + 1e-9, 0.05), 4)
                    grids.append(np.concatenate([g, t2.loc[i] + g]) / gmt)
                L = max(len(g) for g in grids); tg = torch.tensor(np.stack([np.pad(g, (0, L - len(g)), mode='edge') for g in grids]), dtype=torch.float32)
                pred, _ = make_predictions(tg, t_dense, k, f, eb['reducer'], gm, gs)
                pred = pred.numpy()
                for j, i in enumerate(idl):
                    tau = 12.0 if info.loc[i].form == 1 else 24.0; n = len(np.arange(0, tau + 1e-9, 0.05)); g = np.round(np.arange(0, tau + 1e-9, 0.05), 4)
                    res['id'].append(int(i)); res['true_auc_v1'].append(float(auc.loc[(i, 1)])); res['true_auc_v2'].append(float(auc.loc[(i, 2)]))
                    res['pred_auc_v1'].append(float(np.trapezoid(pred[j, :n], g))); res['pred_auc_v2'].append(float(np.trapezoid(pred[j, n:2 * n], g)))
        a1 = np.abs(np.array(res['pred_auc_v1']) / np.array(res['true_auc_v1']) - 1); a2 = np.abs(np.array(res['pred_auc_v2']) / np.array(res['true_auc_v2']) - 1)
        name = os.path.basename(P_te).replace('.csv', '')
        json.dump(dict(model='EB-VAE (Baaz 2026) + patches P1-P3', seed=seed, test=P_te, per_patient=res,
                       v1=dict(median_abs=100 * float(np.median(a1))), v2=dict(median_abs=100 * float(np.median(a2)))), open(f'{out}/eval_{name}_seed{seed}_ep{E}.json', 'w'))
        print(f'epoch {E} {name}: {len(a1)} patients, median |AUC err| V1 {100 * np.median(a1):.1f} %, V2 {100 * np.median(a2):.1f} %', flush=True)
    torch.set_rng_state(rng[0]); np.random.set_state(rng[1]); random.setstate(rng[2])
prev = 0
for E in CKPTS:
    train_loop_model(models_norm=med, models_eval=eb, dataloader=tl, dataset_train=df, main_params=p2, optimizer=o2, scheduler=s2,
                     t_dense=t_dense, n_epochs=E - prev, warmup_epochs_noise=max(0, 200 - prev), warmup_epochs_iiv=0, smoothing_start_epoch=0,
                     enable_ae=False, enable_vae=True, enable_onlymedian=False, normalization=True, truncation=1, print_epoch=10)
    prev = E
    print(f'stage 2 epoch {E} done {(time.time() - t0) / 60:.1f} min', flush=True)
    torch.save({'med': {k: m.state_dict() for k, m in med.items()}, 'eb': {k: m.state_dict() for k, m in eb.items()}}, f'{out}/models_seed{seed}_ep{E}.pt')
    evaluate(E)
print(f'TACRO_BAAZ_DONE {(time.time() - t0) / 60:.1f} min')
