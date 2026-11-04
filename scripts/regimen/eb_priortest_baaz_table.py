#!/usr/bin/env python3
"""Prior test on the design of Baaz et al.: does the EB-VAE's dose-conditioned prior MEAN carry its flattened dose response? Sparse sampling, linear simulator,
randomised arms (seeds 0-4) and titrated arms (rho 0.8, seeds 0-2). Control = the Table 3 runs (res_titr/{ctrl,titr0.8}); test = the
same runs with BAAZ_PRIOR_NO_DOSE=1 (prior mean ignores the first amount, both training stages; res_titr/prior_nodose/...). Paired by
seed (same data, same initialisation stream). Metrics as baaz_arm_split.py / make_fig_ebvae_vs_ours.py: unseen = mean of the median
|AUC 0-30 h err| over the 5 unseen regimens; model/true = median over test patients of the predicted AUC(850)/AUC(350) ratio divided
by the true one; cross-arm = median signed error observed on 800 -> predicted at 350 | observed on 400 -> predicted at 850.
Second test: BAAZ_ENC_NORM=dose -> the stage-2 encoder input is
the concentrations divided by the patient's first amount (relative to 800 mg) and the global mean concentration, as our dose-normalised
encoder, INSTEAD of the concentrations divided by the stage-1 population prediction under the patient's own regimen; prior unchanged
(dose-conditioned) -> res_titr/encnorm_dose/...
Reproducibility: control sparse s1 rerun unchanged (res_titr/repro/ctrl) vs the original.
PRE-STATED (before the runs): the mechanism predicts that under titration the dose-free prior raises model/true by >= 0.15 (from 0.47)
and lowers the 800 -> 350 error by >= 20 points (from +72.5); |change| < 0.05 in model/true in both designs = the prior mean is not the
carrier on this design either; in between = partial. Changes must exceed the run-to-run difference of the reproducibility rerun.
PRE-STATED for the second test (before its runs), same thresholds: under titration the dose-normalised encoder raises model/true by
>= 0.15 AND lowers 800 -> 350 by >= 20 pt -> the population-prediction normalisation is a substantial carrier; |change| < 0.05 in both
designs -> it is not; in between -> partial.
Writes results/regimen_dh/eb_priortest_baaz_table.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, json
import numpy as np

EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen/res_titr'
UNSEEN = ('amt350', 'amt600', 'amt850', 'time400', 'time800')
SEEDS = {'ctrl': range(5), 'titr0.8': range(3)}


def cell(path, truth_path):
    if not os.path.exists(path):
        return None
    r = json.load(open(path))['individual']; t = json.load(open(truth_path))['individual']
    tru = {g: np.array(t[g]['auc_pred_per_patient']) / (1 + np.array(t[g]['auc_err_per_patient'])) for g in ('amt350', 'amt850')}
    e = {g: np.array(r[g]['auc_err_per_patient']) for g in ('amt350', 'amt850')}
    u = float(np.mean([r[g]['auc_abs_median'] for g in UNSEEN]))
    q = (1 + e['amt850']) / (1 + e['amt350'])
    return dict(unseen=u, train=float(r['train']['auc_abs_median']), rel=float(np.median(q)),
                a3=100 * float(np.median(e['amt350'][1::2])), a8=100 * float(np.median(e['amt850'][0::2])),
                raw=np.array(r['amt350']['auc_err_per_patient']))


out = [__doc__.split('\n')[0], '']
for cond, lab in (('ctrl', 'randomised arms'), ('titr0.8', 'TITRATED arms')):
    out.append(f'=== sparse, {lab}, seeds {list(SEEDS[cond])}')
    out.append(f'   {"":22s}{"unseen":>14s}{"training":>14s}{"model/true":>20s}{"800->350 | 400->850":>24s}   per seed model/true')
    C, V = {}, {'prior_nodose': {}, 'encnorm_dose': {}}
    for s in SEEDS[cond]:
        ctrl = f'{EB}/{cond}/N100_sparse_s{s}.json'
        C[s] = cell(ctrl, ctrl)
        for v in V:
            V[v][s] = cell(f'{EB}/{v}/{cond}/N100_sparse_s{s}.json', ctrl)
    for name, D in (('EB-VAE (Table 3)', C), ('EB-VAE, prior w/o dose', V['prior_nodose']), ('EB-VAE, encoder / dose', V['encnorm_dose'])):
        ok = {s: v for s, v in D.items() if v is not None}
        if not ok:
            out.append(f'   {name:22s}  (not run yet)'); continue
        g = lambda k: [v[k] for v in ok.values()]
        msd = lambda v: f'{np.mean(v):.1f} +- {np.std(v, ddof=1) if len(v) > 1 else 0:.1f}'
        out.append(f'   {name:22s}{msd(g("unseen")):>14s}{msd(g("train")):>14s}'
                   + f'{np.mean(g("rel")):.2f} [{min(g("rel")):.2f}-{max(g("rel")):.2f}]'.rjust(20)
                   + f'{np.mean(g("a3")):+11.1f} | {np.mean(g("a8")):+7.1f}   ' + ' '.join(f's{s} {v["rel"]:.2f}' for s, v in sorted(ok.items())))
    for v, vlab in (('prior_nodose', 'prior w/o dose'), ('encnorm_dose', 'encoder / dose')):
        T = V[v]
        S = [s for s in SEEDS[cond] if C[s] is not None and T[s] is not None]
        if S:
            d = lambda k: np.array([T[s][k] - C[s][k] for s in S])
            out.append(f'   paired ({vlab} - Table 3), {len(S)} seeds: model/true {d("rel").mean():+.2f} (' + ' '.join(f'{x:+.2f}' for x in d('rel'))
                       + f'), 800->350 {d("a3").mean():+.1f} pt, unseen {d("unseen").mean():+.1f} pt (lower in {int((d("unseen") < 0).sum())}/{len(S)})')
    out.append('')
rep, orig = f'{EB}/repro/ctrl/N100_sparse_s1.json', f'{EB}/ctrl/N100_sparse_s1.json'
if os.path.exists(rep):
    a, b = cell(rep, orig), cell(orig, orig)
    out.append(f'reproducibility (control sparse s1 rerun unchanged): model/true {a["rel"]:.3f} vs {b["rel"]:.3f}, unseen {a["unseen"]:.2f} vs '
               f'{b["unseen"]:.2f}; max |diff| of per-patient AUC errors at 350 mg {np.abs(a["raw"] - b["raw"]).max():.2e}')
else:
    out.append('reproducibility rerun: not run yet')
os.makedirs('results/regimen_dh', exist_ok=True)
open('results/regimen_dh/eb_priortest_baaz_table.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
