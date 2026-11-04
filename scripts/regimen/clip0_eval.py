#!/usr/bin/env python3
"""Non-negative predictions by clipping at zero (; a
readout trained through a ReLU broke tacrolimus training, pilot_table.txt). No retraining: each run is rebuilt (dose-event, dose input:
dh_regimen.py as a module with the run's flags) or its dumped curves are read (EB-VAE), predictions max(pred, 0), median |AUC err| per
regimen recomputed. CHECK: the unclipped recomputation reproduces each run's JSON (1e-2 points).
    clip0_eval.py [models=de,eb,dc,luis,lunmi]   (de = dose-event, eb = EB-VAE, dc = dose input (ablation), luis / lunmi = Lu et al.;
                                                   de_titr = dose-event, titrated saturable sparse s0-2)   -> results/regimen_dh/clip0/<run tag>.json (per regimen: unclipped, clipped, share of patients < 0)
                                      and results/regimen_dh/clip0/summary.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, runpy, io, contextlib
import numpy as np, torch
ROOT = REPO; os.chdir(ROOT)
SCR = os.environ.get('CLIP_SCRATCH', 'results/scratch/clip0_scratch'); os.makedirs(SCR, exist_ok=True)
OUT = 'results/regimen_dh/clip0'; os.makedirs(OUT, exist_ok=True)
R = 'results/regimen_dh'; EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'
WHICH = (sys.argv[1] if len(sys.argv) > 1 else 'de,eb,dc,luis,lunmi').split(',')
DESIGNS = (('sparse', range(5)), ('visit', range(5)), ('dense', range(3)))
# 22/10 opt-ins (review, job 1 / job 3), defaults unchanged: CLIP_DESIGNS='visit1:5' (design:n_seeds,...), CLIP_DE_SUB / CLIP_DC_SUB = run
# directories of the dose-event model / dose-input ablation, CLIP_LUIS_PROTOCOL=factual (encoder on the observed regimen, as Lu et al.'s
# code predicts a counterfactual regimen), CLIP_TAG_SFX appended to every output tag.
if os.environ.get('CLIP_DESIGNS'):
    DESIGNS = tuple((x.split(':')[0], range(int(x.split(':')[1]))) for x in os.environ['CLIP_DESIGNS'].split(','))
SFX = os.environ.get('CLIP_TAG_SFX', '')
LUIS_PROTOCOL = os.environ.get('CLIP_LUIS_PROTOCOL', 'target'); assert LUIS_PROTOCOL in ('target', 'factual')
DC_SUB = os.environ.get('CLIP_DC_SUB', 'dc_tuned_z12')
CELLS = [(sc, d, s) for sc in (2, 4) for d, ss in DESIGNS for s in ss]
DE = {2: f'{R}/tacro_s2/Bfixkp3.1n', 4: f'{R}/tacro/Bfixkp4n'}
if os.environ.get('CLIP_DE_SUB'):
    DE = {2: f"{R}/tacro_s2/{os.environ['CLIP_DE_SUB']}", 4: f"{R}/tacro/{os.environ['CLIP_DE_SUB']}"}
EBD = {2: f'{EB}/res_tacro_s2/ctrl', 4: f'{EB}/res_tacro/ctrl'}
SUB = {2: f'{R}/tacro_s2', 4: f'{R}/tacro'}   # dc_tuned_z12 / luis / lunmi below these
torch.set_num_threads(1)


def rebuild(pt, sc):
    ck = torch.load(pt, map_location='cpu', weights_only=False); c = ck['cli']
    argv = ['x', str(c['n_per_arm']), c['design'], str(c['seed']), f'{SCR}/u.json', '--no-check', '--iters', '1', '--enc-dose', c['enc_dose'], '--sim', 'tacro']
    argv += ['--obsrv-mgl', str(c['obsrv_mgl'])] if c.get('obsrv_mgl') is not None else []
    argv += ['--kick-phys', str(c['kick_phys'])] if c.get('kick_phys') else []
    argv += ['--titrate', str(c['titrate'])] if c.get('titrate') else []
    for k in ('dose_cond', 'dc_zero_init', 'plain'):
        argv += ['--' + k.replace('_', '-')] if c.get(k) else []
    argv += ['--latents', str(c['latents'])] if c.get('latents') else []
    argv += ['--readout-softplus', str(c['readout_softplus'])] if c.get('readout_softplus') else []
    assert not c.get('depot') and not c.get('readout_relu')
    os.environ['TACRO_SCENARIO'] = str(sc); sys.argv = argv
    for mod in ('gen_tacro_regimens', 'gen_regimens'):   # the simulator reads TACRO_SCENARIO at import
        sys.modules.pop(mod, None)
    with contextlib.redirect_stdout(io.StringIO()):
        g = runpy.run_path('scripts/regimen/dh_regimen.py', run_name='clip0')
    g['model'].load_state_dict(ck['state_dict']); g['model'].eval()
    return g


def lu_preds(pt, sc, REG):
    ck = torch.load(pt, map_location='cpu', weights_only=False); c = ck['cli']
    os.environ['TACRO_SCENARIO'] = str(sc)
    for mod in ('gen_tacro_regimens', 'gen_regimens'):
        sys.modules.pop(mod, None)
    sys.argv = ['x', c['model'], '100', c['design'], str(c['seed']), f'{SCR}/u.json', '--sim', 'tacro'] + \
               (['--euler-dt', repr(c['euler_dt'])] if c.get('euler_dt') not in (None, 0.25) else []) + \
               (['--nmi-x0', c['nmi_x0']] if c.get('nmi_x0', 'encoded') != 'encoded' else [])
    with contextlib.redirect_stdout(io.StringIO()):
        h = runpy.run_path('scripts/regimen/lu_regimen.py', run_name='clip0')
    m = h['model']; m.load_state_dict(ck['state_dict']); m.eval(); out = {}
    with torch.no_grad():
        if c['model'] == 'is' and LUIS_PROTOCOL == 'factual':   # encoder on the observed (training) regimen, new doses in the rollout
            z0, _ = m.encode(h['feats_te'])
            for r in REG:
                times, A = h['regs'][r]
                out[r] = m.decode(m.rollout(z0, times, A, h['EG']), h['ctx_te']).numpy() * h['YMAX']
        elif c['model'] == 'is':   # as published: the encoder reads the samples with the dosing channels of the regimen predicted
            for r in REG:
                times, A = h['regs'][r]
                z0, _ = m.encode(h['rows'](h['y_te'], times, A, cov=h['covs'](h['P_te']))[0])
                out[r] = m.decode(m.rollout(z0, times, A, h['EG']), h['ctx_te']).numpy() * h['YMAX']
        else:
            theta, x0 = m.encode(h['feats_te'])
            for r in REG:
                out[r] = m.simulate(theta, x0, *h['regs'][r], h['EG']).numpy() * h['YMAX']
    return out, h['P_te'], h['EG']


def score(pred, truth, EG):
    return 100 * float(np.median(np.abs(np.trapezoid(pred, EG, axis=1) / np.trapezoid(truth, EG, axis=1) - 1)))


def evaluate(tag, preds, truth, EG, ref):
    res = {}
    for r in preds:
        p, t = preds[r], truth[r]
        e0 = score(p, t, EG)
        assert abs(e0 - ref[r]['auc_abs_median']) < 1e-2, f'{tag} {r}: recomputed {e0:.3f} vs run {ref[r]["auc_abs_median"]:.3f}'
        neg = np.trapezoid(np.clip(-p, 0, None), EG, axis=1) / np.trapezoid(np.abs(p), EG, axis=1)
        res[r] = dict(unclipped=e0, clipped=score(np.clip(p, 0, None), t, EG), share_below0=float(np.mean(p.min(1) < 0)),
                      neg_area_median_pct=100 * float(np.median(neg)))
    json.dump(res, open(f'{OUT}/{tag}.json', 'w'), indent=1)
    return res


L = [__doc__.split('\n')[0], '']
for sc, d, s in CELLS:
    for m in WHICH:
        tag = f'{m}_sc{sc}_{d}_s{s}{SFX}'
        if m == 'de_titr' and not (sc == 4 and d == 'sparse' and s < 3): continue
        if m in ('de', 'de_titr'):
            base = f'{DE[sc]}/N100_{d}_s{s}' if m == 'de' else f'{R}/tacro/titr0.8/Bfixkp4n/N100_{d}_s{s}'
            if not os.path.exists(base + '.pt'): continue
            g = rebuild(base + '.pt', sc); EG = g['EG']; reg = g['regs']; ref = json.load(open(base + '.json'))['individual']
            with torch.no_grad():
                mu, _ = g['model'].encoder_z0(g['X_te'], g['tp_samp'], static=g['S_te'], run_backwards=True)
                preds = {r: g['decode'](g['simulate'](mu, *reg[r], EG))[0].numpy() * g['YMAX'] for r in reg}
            truth = {r: np.stack([np.interp(EG, g['t_eval'], cc) for cc in g['truth'][r]]) for r in reg}
            TR = (truth, EG, g['P_te'])   # the other models of the same cell are scored against this truth
        elif m == 'dc':
            base = f'{SUB[sc]}/{DC_SUB}/N100_{d}_s{s}'
            if not os.path.exists(base + '.pt') or TR is None: continue
            g = rebuild(base + '.pt', sc); reg = g['regs']; ref = json.load(open(base + '.json'))['individual']; truth, EG, P0 = TR
            assert np.allclose(g['P_te'], P0) and np.allclose(g['EG'], EG)
            with torch.no_grad():
                mu, _ = g['model'].encoder_z0(g['X_te'], g['tp_samp'], static=g['S_te'], run_backwards=True)
                preds = {r: g['decode'](g['simulate'](mu, *reg[r], EG))[0].numpy() * g['YMAX'] for r in reg}
        elif m in ('luis', 'lunmi'):
            base = f'{SUB[sc]}/{m}/N100_{d}_s{s}'
            if not os.path.exists(base + '.pt') or TR is None: continue
            truth, EG, P0 = TR
            ref = json.load(open(base + '.json'))['individual_factual_encoder' if (m == 'luis' and LUIS_PROTOCOL == 'factual') else 'individual']
            preds, P1, EG1 = lu_preds(base + '.pt', sc, list(truth))
            assert np.allclose(P1, P0) and np.allclose(EG1, EG), f'{m}: other test patients or grid'
        else:
            f = f'{EBD[sc]}/N100_{d}_s{s}'
            if not os.path.exists(f + '_curves.npz') or TR is None: continue
            D = np.load(f + '_curves.npz'); truth, EG, _ = TR; assert np.allclose(D['EG'], EG)
            ref = json.load(open(f + '.json'))['individual']; preds = {r: D[r].astype(float) for r in truth}
        res = evaluate(tag, preds, truth, EG, ref)
        un = [r for r in res if r != 'train']
        L.append(f'{tag:24s} unseen {np.mean([res[r]["unclipped"] for r in un]):5.1f} -> {np.mean([res[r]["clipped"] for r in un]):5.1f}  '
                 f'training {res["train"]["unclipped"]:5.1f} -> {res["train"]["clipped"]:5.1f}  patients < 0 (max over regimens) '
                 f'{100 * max(v["share_below0"] for v in res.values()):3.0f}%  negative area median {max(v["neg_area_median_pct"] for v in res.values()):.2f}%')
        print(L[-1], flush=True)
    TR = None
open(f'{OUT}/summary' + ('' if WHICH == ['de', 'eb', 'dc', 'luis', 'lunmi'] else '_' + '_'.join(WHICH)) + SFX + '.txt', 'w').write('\n'.join(L) + '\n')
