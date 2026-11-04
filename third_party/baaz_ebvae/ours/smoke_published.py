"""Smoke test of the published EB-VAE pipeline (Baaz et al. 2026, MIT) on its OWN simulated design, reduced:
correlated 1-cpt-with-absorption PK, 400 / 800 mg x4 doses (0, 8, 16, 24 h), n per arm small; stage 1 = 'median'
model (k fixed at the prior mean), stage 2 = EB-VAE with input-response normalisation. Same calls and settings as
scripts/main_simulated_data.py, which cannot run as shipped (it loads a pre-trained median model from models/).
Run from third_party/baaz_ebvae:  MPLBACKEND=Agg python ours/smoke_published.py <n_per_arm> <epochs_med> <epochs_vae>"""
import os, sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, os.getcwd())
from lib.utils.utils_data_generation import simulate_single_drug_concentration
from lib.utils.utils_preprocess import prepare_optimizer, prepare_datasets_and_loaders_simulated, compute_global_stats
from lib.utils.utils_training import train_loop_model
from lib.models.NNmodels import Encoder_Transformer, ODEFunc, SimpleDecoder, TrainableNoise
n, ep_med, ep_vae = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
D = 'ours/smoke_data'; os.makedirs(D, exist_ok=True); torch.manual_seed(0); np.random.seed(0)
corr = np.array([[1.0, 0.3, 0.35], [0.3, 1.0, 0.2], [0.35, 0.2, 1.0]])
def make(path, n, seed):
    np.random.seed(seed); parts, off = [], 0
    for gi, amt in enumerate((400, 800), start=1):
        g = simulate_single_drug_concentration(n_individuals=n, dose_amounts=[amt] * 4, dose_times=[0, 8, 16, 24],
            ka_mean=0.4, ke_mean=0.6, v_mean=50, ka_sd=0.5, ke_sd=0.5, v_sd=0.5, add_e=10, prop_e=0.0001,
            t_integration=(0, 30), t_interval=(0, 30), sample_frequency=0.5, save_path=f'{D}/tmp.csv', plot=False, corr_matrix=corr)
        g.columns = g.columns.str.strip().str.upper(); g['ID'] += off; off = g['ID'].max() + 1; g['TREATMENT'] = gi; parts.append(g)
    pd.concat(parts).sort_values(['ID', 'TIME']).to_csv(path, index=False, sep=';')
t0 = time.time()
for nm, s in (('train', 1), ('val', 2), ('test', 3)): make(f'{D}/{nm}.csv', n, s)
df = pd.read_csv(f'{D}/train.csv', sep=';'); print('columns:', list(df.columns)); print(df.head(8).to_string())
gmd, gmt, gm, gs, gmax, gmin = compute_global_stats(df)
out = prepare_datasets_and_loaders_simulated(f'{D}/train.csv', f'{D}/val.csv', f'{D}/test.csv', gm, gmd, gmt, gm, gs, gs,
                                             torch.device('cpu'), batch_fraction=0.05, time_points=120)
ds_train, ds_val, ds_test, base, tl, vl, tsl, t_dense = out[:8]
def build(learn_cov, h_enc, m_enc, layers, add_std):
    enc = Encoder_Transformer(dim_latent=2, dim_parameters_IC=1, dim_parameters_dynamic=2, dim_cov=1, model_dim=m_enc, hidden_dim=h_enc,
        num_heads=2, num_layers=layers, dropout=0.1, cov_diag_epsilon=1e-5, learn_prior_mean=True, learn_prior_covariance=learn_cov,
        diagonal_only=False, IC_dose_dependent=False)
    return {'func': ODEFunc(dim_latent=2, dim_parameters_IC=1, dim_parameters_dynamic=2, hid_dim=32, drug_dim=1), 'encoder': enc,
            'reducer': SimpleDecoder(2), 'noise': TrainableNoise(init_add_std=add_std, init_prop_std=0.01)}
med = build(False, 8, 16, 1, 0.2)
o1, s1, p1 = prepare_optimizer(med, torch.device('cpu'), lr=1e-3, factor=0.8, patience=20, min_lr=1e-5, prior_lr_factor=1.0)
train_loop_model(med, med, dataloader=tl, dataset_train=df, main_params=p1, optimizer=o1, scheduler=s1, t_dense=t_dense, n_epochs=ep_med,
                 warmup_epochs_noise=1000, warmup_epochs_iiv=0, smoothing_start_epoch=0, enable_ae=False, enable_vae=False,
                 enable_onlymedian=True, normalization=False, truncation=1)
print(f'stage 1 (median model) done, {time.time() - t0:.0f}s', flush=True)
eb = build(True, 16, 32, 2, 0.1)
o2, s2, p2 = prepare_optimizer(eb, torch.device('cpu'), lr=1e-3, factor=0.8, patience=10, min_lr=1e-5, prior_lr_factor=5.0)
train_loop_model(models_norm=med, models_eval=eb, dataloader=tl, dataset_train=df, main_params=p2, optimizer=o2, scheduler=s2, t_dense=t_dense,
                 n_epochs=ep_vae, warmup_epochs_noise=200, warmup_epochs_iiv=0, smoothing_start_epoch=0, enable_ae=False, enable_vae=True,
                 enable_onlymedian=False, normalization=True, truncation=1)
print(f'SMOKE_OK stage 2 (EB-VAE) done, {time.time() - t0:.0f}s')
