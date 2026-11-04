#!/usr/bin/env python3
"""Latent test: does the encoded patient depend on the dose the patient happened to receive? Design of Baaz et al., linear simulator, sparse
sampling, randomised (s0-4) and titrated (rho 0.8, s0-2) training. The SAME 400 test patients are observed under the training schedule at
400 mg, again at 400 mg (fresh noise) and at 800 mg (EB-VAE runner with BAAZ_DUMP_LATENT=1, stream seed + 9091; res_titr/latent/...), and
encoded by
  EB-VAE   its own pipeline (population-model normalisation, AE-mode posterior mean k, 4 dims) -- retrained run, deterministic: CHECK
           its per-patient test errors equal the Table 3 run's
  ours     the Table 3 dose-event model (results/regimen_dh/[titr0.8/]Bfixkp20n/N100_sparse_s*.pt; rebuilt with dh_regimen.py, run as a
           module, no training), posterior mean z0 (10 dims) and its physiology coordinates p = Q_p^T z0 (4 dims); CHECK same P_te, SAMP
Measures (scale-free; rows = patients; reference spread = 2 tr Cov(e_400a)):
  arm move     E|e_800 - e_400a|^2 / 2 tr Cov    (0 = same patient whatever the dose, 1 = as different as two random patients)
  noise floor  E|e_400b - e_400a|^2 / 2 tr Cov    (same dose, fresh observation noise)
  common share |mean(e_800 - e_400a)|^2 / E|e_800 - e_400a|^2   (part of the arm move shared by all patients = a systematic shift)
  read-out     5-fold ridge R^2 of the true log ka / log ke / log V from e_400a; 'same map at 800 mg' = the read-out fitted on e_400a
               applied to e_800 (can one read-out read the patient whatever the dose?)
PRE-STATED (before the runs): ours arm move - noise floor < 0.05 and the 800-mg read-out of log ke within 0.05 of the 400-mg one; EB-VAE
arm move above its noise floor by more than ours, larger under titration than with randomised arms, with a lower 800-mg read-out.
Writes results/regimen_dh/latent_dose_invariance.{txt,npz}."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, runpy
import numpy as np, torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

ROOT = os.getcwd(); EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen/res_titr'
SCR = os.environ.get('LATENT_SCRATCH', 'results/scratch/latent_scratch'); os.makedirs(SCR, exist_ok=True)
SEEDS = {'ctrl': range(5), 'titr0.8': range(3)}
torch.set_num_threads(1)


def ridge():
    return Pipeline([('s', StandardScaler()), ('m', RidgeCV(alphas=np.logspace(-3, 3, 13)))])


def measures(e, y):
    a, b, c = e['400a'], e['400b'], e['800']
    den = 2 * np.trace(np.cov(a.T))
    d = c - a
    out = dict(arm=float(np.mean(np.sum(d ** 2, 1)) / den), noise=float(np.mean(np.sum((b - a) ** 2, 1)) / den),
               common=float(np.sum(d.mean(0) ** 2) / np.mean(np.sum(d ** 2, 1))))
    for name, t in y.items():
        p400 = np.zeros_like(t); p800 = np.zeros_like(t)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(a):
            m = ridge().fit(a[tr], t[tr]); p400[te] = m.predict(a[te]); p800[te] = m.predict(c[te])
        out[f'r2_{name}'] = float(r2_score(t, p400)); out[f'r2x_{name}'] = float(r2_score(t, p800))
    return out


def ours_encode(pt, D):
    ck = torch.load(pt, map_location='cpu', weights_only=False); cli = ck['cli']
    argv = ['x', str(cli['n_per_arm']), cli['design'], str(cli['seed']), f'{SCR}/unused.json', '--no-check', '--iters', str(cli['iters'])]
    if cli.get('obsrv_mgl') is not None:
        argv += ['--obsrv-mgl', str(cli['obsrv_mgl'])]
    for k, flag in (('kick_phys', '--kick-phys'), ('latents', '--latents'), ('split', '--split'), ('enc_dose', '--enc-dose'), ('titrate', '--titrate')):
        if cli.get(k) is not None:
            argv += [flag, str(cli[k])]
    sys.argv = argv
    g = runpy.run_path(f'{ROOT}/scripts/regimen/dh_regimen.py', run_name='latent')
    g['model'].load_state_dict(ck['state_dict']); g['model'].eval()
    assert np.allclose(g['P_te'], D['P_te']), 'test patients differ between the two runners'
    assert np.allclose(g['SAMP'], D['samp']), 'sampling times differ'
    os_ = g['model'].orth_split; Q = os_.Q().detach(); kfd = os_.k_f + os_.k_d
    z, p = {}, {}
    with torch.no_grad():
        for tag, a_mg in (('400a', 400.0), ('400b', 400.0), ('800', 800.0)):
            A = np.full((len(D['P_te']), 4), a_mg)
            X, S = g['enc_inputs'](D[f'y_{tag}'], A, g['TRAIN_TIMES'], g['covs'](g['P_te']))
            mu, _ = g['model'].encoder_z0(X, g['tp_samp'], static=S, run_backwards=True)
            z[tag] = mu[0].numpy(); p[tag] = (mu[0] @ Q[:, kfd:]).numpy()
    return z, p


lines, store = [__doc__.split('\n')[0], ''], {}
for cond, lab in (('ctrl', 'randomised arms'), ('titr0.8', 'TITRATED arms')):
    lines.append(f'=== sparse, {lab}')
    lines.append(f'   {"seed":5s}{"model":22s}{"arm move":>9s}{"noise":>8s}{"excess":>8s}{"common":>8s}   R2 log ka / ke / V at 400 mg   (same map at 800 mg)')
    rows = {'EB-VAE (k)': [], 'ours (z0)': [], 'ours (physiology p)': []}
    for s in SEEDS[cond]:
        npz = f'{EB}/latent/{cond}/N100_sparse_s{s}_latent.npz'
        if not os.path.exists(npz):
            lines.append(f'   s{s}   (EB-VAE latent dump not available yet)'); continue
        D = dict(np.load(npz))
        new, old = json.load(open(f'{EB}/latent/{cond}/N100_sparse_s{s}.json')), json.load(open(f'{EB}/{cond}/N100_sparse_s{s}.json'))
        same = all(np.allclose(new['individual'][g]['auc_err_per_patient'], old['individual'][g]['auc_err_per_patient'], atol=0, rtol=0)
                   for g in old['individual'])
        y = {'ka': np.log(D['P_te'][:, 0]), 'ke': np.log(D['P_te'][:, 1]), 'V': np.log(D['P_te'][:, 2])}
        pt = f'{ROOT}/results/regimen_dh/{"titr0.8/" if cond == "titr0.8" else ""}Bfixkp20n/N100_sparse_s{s}.pt'
        z, p = ours_encode(pt, D)
        k = {t: D[f'k_{t}'] for t in ('400a', '400b', '800')}
        for name, e in (('EB-VAE (k)', k), ('ours (z0)', z), ('ours (physiology p)', p)):
            m = measures(e, y); rows[name].append(m); store[f'{cond}_s{s}_{name}'] = np.array([m[x] for x in sorted(m)])
            lines.append(f'   s{s:<4d}{name:22s}{m["arm"]:9.3f}{m["noise"]:8.3f}{m["arm"] - m["noise"]:8.3f}{m["common"]:8.2f}   '
                         f'{m["r2_ka"]:.2f} / {m["r2_ke"]:.2f} / {m["r2_V"]:.2f}   ({m["r2x_ka"]:.2f} / {m["r2x_ke"]:.2f} / {m["r2x_V"]:.2f})'
                         + ('' if name != 'EB-VAE (k)' else f'   [retrained run identical to Table 3: {same}]'))
    for name, R in rows.items():
        if R:
            f = lambda key: np.mean([r[key] for r in R])
            lines.append(f'   mean {name:22s}{f("arm"):9.3f}{f("noise"):8.3f}{f("arm") - f("noise"):8.3f}{f("common"):8.2f}   '
                         f'{f("r2_ka"):.2f} / {f("r2_ke"):.2f} / {f("r2_V"):.2f}   ({f("r2x_ka"):.2f} / {f("r2x_ke"):.2f} / {f("r2x_V"):.2f})')
    lines.append('')
os.makedirs(f'{ROOT}/results/regimen_dh', exist_ok=True)
open(f'{ROOT}/results/regimen_dh/latent_dose_invariance.txt', 'w').write('\n'.join(lines) + '\n')
np.savez_compressed(f'{ROOT}/results/regimen_dh/latent_dose_invariance.npz', **store)
print('\n'.join(lines))
