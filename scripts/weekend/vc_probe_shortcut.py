#!/usr/bin/env python3
"""Re-check of the Note S6 statement "the central volume, recoverable from the encoder's latent state with R^2 ~ 0.86" (project notes
0.00zz88/0.00zz92: the same kind of probe gave 0.80 on the main cohort, of which the formulation alone explained 0.78). Shortcut test
of step 2 (confounding by the central volume, windowed cohorts, 200 training patients): the CONTROL-trained models (cohort
confound_vc00_win_n200, results/vcw200_matched/s<seed>/, ep6000), OT-FiLM and the dose-conditioned field, seeds 1-10. Test patients of
confound_vc00_win (not in its training split), normalisation constants of the training cohort, visit-1 posterior mean z0.
5-fold ridge R^2 of log Vc (truth: confound_truth.csv) from z0: all patients, within Prograf, within Advagraf; and from the formulation
alone (indicator) for reference. No training. Writes results/vcw200_matched/vc_probe_within_formulation.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, glob
import numpy as np, pandas as pd, torch
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
os.chdir(REPO); sys.path.insert(0, os.getcwd())
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film, set_static_hematocrit
from probe_entanglement import load_film_model, load_dosecond_model, encode_mu
torch.set_num_threads(1); dev = torch.device('cpu')

B, T, RE = 'confound_vc00_win', 'confound_vc00_win_n200', './results/exp_film_run'
tru = pd.read_csv(f'{RE}/{B}/confound_truth.csv').set_index('ID')
assert (tru.confound_param == 'Vc').all()


def r2(X, y):
    m = Pipeline([('s', StandardScaler()), ('r', RidgeCV(alphas=np.logspace(-3, 3, 13)))])
    return float(r2_score(y, cross_val_predict(m, X, y, cv=KFold(5, shuffle=True, random_state=0))))


L = [__doc__.split('\n')[0], '', f'{"model":10s}{"seed":>5s}{"n":>6s}{"all":>8s}{"Prograf":>9s}{"Advagraf":>10s}{"formulation only":>18s}']
res = {'film': [], 'dosecond': []}
for s in range(1, 11):
    for arch, sub, loader in (('film', 'exp_film_run', load_film_model), ('dosecond', 'exp_dosecond_run', load_dosecond_model)):
        ck = glob.glob(f'results/vcw200_matched/s{s}/{sub}/{T}/traj/*_ep006000.ckpt')
        if len(ck) != 1:
            L.append(f'{arch:10s}{s:5d}   checkpoint missing'); continue
        a = torch.load(ck[0], map_location='cpu', weights_only=False)['args']
        set_static_hematocrit(int(getattr(a, 'static_dim', 3)) >= 4)
        model, _ = loader(ck[0], dev); model.eval()
        pin = extract_gen_tac_film(file_path=[f'{RE}/{T}/virtual_cohort_film_train.csv'])[1]
        tr, te = (f'{RE}/{B}/virtual_cohort_film_{x}.csv' for x in ('train', 'test'))
        data, _ = extract_gen_tac_film(file_path=[tr, te], scale=pin)
        tr_ids = set(extract_gen_tac_film(file_path=[tr], scale=pin)[0].keys())
        ids = [k for k in data if k not in tr_ids]
        b = collate_fn_tacro_film([TacroFilmDataset({k: data[k] for k in ids})[i] for i in range(len(ids))], dev)
        with torch.no_grad():
            z = encode_mu(model, b['observed_data_v1'], b['observed_tp_v1'], b['dose_v1'], b['static_v1']).numpy()
        t = tru.loc[[int(k) for k in ids]]
        y = np.log(t.confound_par.values); prog = (t.DRUG.values == 'Prograf')
        row = dict(all=r2(z, y), prog=r2(z[prog], y[prog]), adv=r2(z[~prog], y[~prog]),
                   form=float(r2_score(y, LinearRegression().fit(prog[:, None].astype(float), y).predict(prog[:, None].astype(float)))))
        res[arch].append(row)
        L.append(f'{arch:10s}{s:5d}{len(ids):6d}{row["all"]:8.2f}{row["prog"]:9.2f}{row["adv"]:10.2f}{row["form"]:18.2f}')
L.append('')
for arch, rows in res.items():
    if rows:
        f = lambda k: np.mean([r[k] for r in rows])
        L.append(f'mean {arch:10s} ({len(rows)} seeds): all {f("all"):.2f}, within Prograf {f("prog"):.2f}, within Advagraf {f("adv"):.2f}, '
                 f'formulation alone {f("form"):.2f}')
open('results/vcw200_matched/vc_probe_within_formulation.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
