#!/usr/bin/env python3
"""Data for the main-text patient figure of ndm.tex (
for the dose-event vs EB-VAE vs Lu for the 6 samples linear tacrolimus, just so people have an understanding of what we are doing
exactly"). Tacrolimus regimen benchmark, LINEAR elimination (scenario 2), sparse sampling (6 samples), one training seed. No training:
  dose-event   results/regimen_dh/tacro_s2/Bfixkp3.1n/N100_sparse_s{seed}.pt, rebuilt with dh_regimen.py (run as a module, the run's flags)
  Lu iScience  results/regimen_dh/tacro_s2/luis/...pt, lu_regimen.py; as published, the encoder reads the regimen to be predicted
  Lu NMI       results/regimen_dh/tacro_s2/lunmi/...pt, lu_regimen.py; patient parameters from the observed samples, new doses through
               the dose port
  EB-VAE       third_party/baaz_ebvae/ours/regimen/res_tacro_s2/ctrl/N100_sparse_s{seed}_curves.npz (curves dumped by the run)
CHECKS: every model's reloaded / dumped curves reproduce the run's own median |AUC(0-45 h) err| on all six regimens (1e-2 points); the
test patients of the two runners are identical; the EB-VAE's per-patient AUC errors recomputed from its curves against our truth equal
the ones in its JSON (same patients, same truth).
Patient (rule fixed before plotting): the test patient of the 2-mg arm with the median true AUC on its training regimen.
    fig_tacro_patients_data.py [design=sparse] [seed=0]  -> results/ndm/fig_tacro_patients[_visit].{npz,txt}
19/10: Fig. 6 with the 12 samples instead of the 6): design argument (sparse | visit = intensive PK visit, 12 samples); the
NMI model is rebuilt with its run's --euler-dt; adds the share of test patients for which the dose-event model has the lower mean
|unseen AUC err| than the EB-VAE, and the curve shape (C(0), time to peak after the last dose) of both models, for the caption."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, runpy, io, contextlib
os.environ['TACRO_SCENARIO'] = '2'
import numpy as np, torch

ROOT = REPO; os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, f'{ROOT}/scripts/regimen')
import gen_tacro_regimens as G
assert G.SCENARIO == 2
DESIGN = sys.argv[1] if len(sys.argv) > 1 else 'sparse'; assert DESIGN in ('sparse', 'visit', 'dense', 'visit1')   # 22/10: visit1 = 12 samples around dose 1   # 21/10: dense = 61 samples (Fig. S2)
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OURS = os.environ.get('FIG_OURS', 'Bfixkp3.1n')   # 19/10: Bdepot3.1n = the depot rerun
CLIP = os.environ.get('FIG_CLIP0') == '1'   # 20/10 : every model's curve max(., 0)
LUIS = os.environ.get('FIG_LUIS', 'target')   # 22/10 opt-in: factual = encoder on the observed regimen (Lu et al.'s code for a counterfactual regimen)
assert LUIS in ('target', 'factual')
SFX = ('' if DESIGN == 'sparse' else f'_{DESIGN}') + ('' if OURS == 'Bfixkp3.1n' else ('_depot' if OURS.startswith('Bdepot') else f'_{OURS}')) \
      + ('_clip0' if CLIP else '') + ('_luisfact' if LUIS == 'factual' else '')
SCR = os.environ.get('FIG_SCRATCH', 'results/scratch/fig_scratch'); os.makedirs(SCR, exist_ok=True)
R = 'results/regimen_dh/tacro_s2'; EBD = f'{REPO}/third_party/baaz_ebvae/ours/regimen/res_tacro_s2/ctrl'
TAG = f'N100_{DESIGN}_s{SEED}'; REG = list(G.REGIMENS)
torch.set_num_threads(1)


def quiet_run(path, argv):
    sys.argv = argv
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(path, run_name='figdata')


def med_abs(pred, tru, eg):
    return 100 * float(np.median(np.abs(np.trapezoid(pred, eg, axis=1) / np.trapezoid(tru, eg, axis=1) - 1)))


pred, log = {}, [__doc__.split('\n')[0], f'design {DESIGN}, seed {SEED}, scenario {G.SCENARIO}, regimens {REG}', '']
# ---- dose-event model (also gives the patients, their samples and the truth)
ck = torch.load(f'{R}/{OURS}/{TAG}.pt', map_location='cpu', weights_only=False); c = ck['cli']
argv = ['x', '100', DESIGN, str(SEED), f'{SCR}/u.json', '--sim', 'tacro', '--enc-dose', c['enc_dose']]
argv += ['--obsrv-mgl', str(c['obsrv_mgl'])] if c.get('obsrv_mgl') is not None else []
argv += ['--kick-phys', str(c['kick_phys'])] if c.get('kick_phys') else []
argv += ['--depot', str(c['depot'])] if c.get('depot') else []
argv += ['--readout-softplus', str(c['readout_softplus'])] if c.get('readout_softplus') else []   # 22/10 (job 1)
g = quiet_run('scripts/regimen/dh_regimen.py', argv)
g['model'].load_state_dict(ck['state_dict']); g['model'].eval()
assert abs(ck['YMAX'] - g['YMAX']) < 1e-9
EG, P_te, arm = g['EG'], g['P_te'], g['A_te'][:, 0]
truth = {r: np.stack([np.interp(EG, g['t_eval'], cc) for cc in g['truth'][r]]) for r in REG}
with torch.no_grad():
    mu, _ = g['model'].encoder_z0(g['X_te'], g['tp_samp'], static=g['S_te'], run_backwards=True)
    pred['dose-event'] = {r: g['decode'](g['simulate'](mu, *g['regs'][r], EG))[0].numpy() * g['YMAX'] for r in REG}
y_obs, samp = g['y_te'], g['SAMP']
runs = {'dose-event': json.load(open(f'{R}/{OURS}/{TAG}.json'))}
# ---- Lu et al.
for key, d in (('Lu iScience', 'luis'), ('Lu NMI', 'lunmi')):
    ck = torch.load(f'{R}/{d}/{TAG}.pt', map_location='cpu', weights_only=False); c = ck['cli']
    h = quiet_run('scripts/regimen/lu_regimen.py', ['x', c['model'], '100', DESIGN, str(SEED), f'{SCR}/u.json', '--sim', 'tacro']
                  + (['--euler-dt', repr(c['euler_dt'])] if c.get('euler_dt') not in (None, 0.25) else [])
                  + (['--nmi-x0', c['nmi_x0']] if c.get('nmi_x0', 'encoded') != 'encoded' else []))
    m = h['model']; m.load_state_dict(ck['state_dict']); m.eval()
    assert abs(ck['YMAX'] - h['YMAX']) < 1e-9 and np.allclose(h['P_te'], P_te), f'{key}: different test patients'
    out = {}
    with torch.no_grad():
        if c['model'] == 'is' and LUIS == 'factual':   # encoder on the observed regimen, new doses added during the rollout
            z0, _ = m.encode(h['feats_te'])
            for r in REG:
                times, A = h['regs'][r]
                out[r] = m.decode(m.rollout(z0, times, A, h['EG']), h['ctx_te']).numpy() * h['YMAX']
        elif c['model'] == 'is':       # as published: the encoder reads the samples with the dosing channels of the regimen predicted
            for r in REG:
                times, A = h['regs'][r]
                z0, _ = m.encode(h['rows'](h['y_te'], times, A, cov=h['covs'](h['P_te']))[0])
                out[r] = m.decode(m.rollout(z0, times, A, h['EG']), h['ctx_te']).numpy() * h['YMAX']
        else:
            theta, x0 = m.encode(h['feats_te'])
            for r in REG:
                out[r] = m.simulate(theta, x0, *h['regs'][r], h['EG']).numpy() * h['YMAX']
    assert np.allclose(h['EG'], EG)
    pred[key] = out; runs[key] = json.load(open(f'{R}/{d}/{TAG}.json'))
    if c['model'] == 'is' and LUIS == 'factual':   # the reload check compares with the run's factual-encoder scores
        runs[key]['individual'] = runs[key]['individual_factual_encoder']
# ---- EB-VAE (dumped curves)
D = np.load(f'{EBD}/{TAG}_curves.npz'); assert np.allclose(D['EG'], EG)
pred['EB-VAE'] = {r: D[r].astype(float) for r in REG}; runs['EB-VAE'] = json.load(open(f'{EBD}/{TAG}.json'))
for r in REG:
    e_ours = np.trapezoid(pred['EB-VAE'][r], EG, axis=1) / np.trapezoid(truth[r], EG, axis=1) - 1
    assert np.allclose(e_ours, runs['EB-VAE']['individual'][r]['auc_err_per_patient'], atol=1e-4), f'EB-VAE {r}: other patients or truth'
# ---- reload checks
MODELS = ['dose-event', 'EB-VAE', 'Lu iScience', 'Lu NMI']
log.append(f'{"model":12s} ' + ' '.join(f'{r:>9s}' for r in REG) + '   (median |AUC 0-45 h err| %, reloaded / run)')
for k in MODELS:
    cells = []
    for r in REG:
        a, b = med_abs(pred[k][r], truth[r], EG), runs[k]['individual'][r]['auc_abs_median']
        assert abs(a - b) < 1e-2, f'{k} {r}: reload {a:.3f} vs run {b:.3f}'
        cells.append(f'{a:9.1f}')
    log.append(f'{k:12s} ' + ' '.join(cells) + f'   unseen mean {np.mean([runs[k]["individual"][r]["auc_abs_median"] for r in G.UNSEEN]):.1f}')
if CLIP:   # after the reload checks, which compare the raw predictions with each run's JSON
    log += ['', 'CLIPPED AT ZERO (every model): share of test patients with a negative prediction on any regimen before clipping: '
            + ', '.join(f'{k} {100 * np.mean(np.any([pred[k][r].min(1) < 0 for r in REG], 0)):.0f}%' for k in MODELS)]
    pred = {k: {r: np.clip(v, 0, None) for r, v in pred[k].items()} for k in MODELS}
    log.append('clipped ' + ', '.join(f'{k} unseen mean {np.mean([med_abs(pred[k][r], truth[r], EG) for r in G.UNSEEN]):.1f}' for k in MODELS))
# ---- the patient
auc_tr = np.trapezoid(truth['train'], EG, axis=1)
low = np.where(np.isclose(arm, G.LOW))[0]
i = int(low[np.argsort(auc_tr[low])[len(low) // 2]])
log += ['', f'patient {i}: arm {arm[i]:g} mg, true AUC(0-45 h) on its training regimen {auc_tr[i]:.1f} (median of the 2-mg arm), '
        f'parameters ' + ', '.join(f'{n} {v:.3g}' for n, v in zip(G.COLS, P_te[i]))]
err = {k: {r: float(np.trapezoid(pred[k][r][i], EG) / np.trapezoid(truth[r][i], EG) - 1) for r in REG} for k in MODELS}
for k in MODELS:
    log.append(f'  {k:12s} AUC err ' + ' '.join(f'{r} {100 * err[k][r]:+.1f}%' for r in REG))
# ---- for the caption: who is more accurate over all test patients, and the shape of the curves
un = lambda k: np.mean([np.abs(np.trapezoid(pred[k][r], EG, axis=1) / np.trapezoid(truth[r], EG, axis=1) - 1) for r in G.UNSEEN], 0)
share = float(np.mean(un('dose-event') < un('EB-VAE')))
w = (EG >= 36.0 - 1e-9) & (EG <= 42.0 + 1e-9)
shape = {k: dict(c0=float(np.median(pred[k]['train'][:, 0])), tpeak=float(np.median(EG[w][np.argmax(pred[k]['train'][:, w], 1)] - 36.0)))
         for k in ('dose-event', 'EB-VAE')}
shape['truth'] = dict(c0=float(np.median(truth['train'][:, 0])), tpeak=float(np.median(EG[w][np.argmax(truth['train'][:, w], 1)] - 36.0)))
log += ['', f'over the {len(P_te)} test patients: dose-event model has the lower mean |unseen AUC err| for {100 * share:.0f}% (EB-VAE for {100 - 100 * share:.0f}%)',
        'curve shape on the training regimen (median over patients): ' + '; '.join(f'{k} C(0) {v["c0"]:.1f} ng/mL, peak {v["tpeak"]:.2f} h after the last dose' for k, v in shape.items())]
os.makedirs('results/ndm', exist_ok=True)
np.savez_compressed(f'results/ndm/fig_tacro_patients{SFX}.npz', EG=EG, samp=samp, y_obs=y_obs[i], patient=i, arm=arm[i], seed=SEED, design=DESIGN, share_ours=share,
                    regimens=np.array(REG), **{f'times_{r}': np.array(g['regs'][r][0], float) for r in REG}, **{f'amounts_{r}': np.array(g['regs'][r][1][i], float) for r in REG},
                    **{f'truth_{r}': truth[r][i] for r in REG},
                    **{f'pred_{k}_{r}': pred[k][r][i] for k in MODELS for r in REG},
                    **{f'err_{k}_{r}': err[k][r] for k in MODELS for r in REG},
                    **{f'medunseen_{k}': (np.mean([med_abs(pred[k][r], truth[r], EG) for r in G.UNSEEN]) if CLIP
                                          else np.mean([runs[k]['individual'][r]['auc_abs_median'] for r in G.UNSEEN])) for k in MODELS})
open(f'results/ndm/fig_tacro_patients{SFX}.txt', 'w').write('\n'.join(log) + '\n')
print('\n'.join(log))
