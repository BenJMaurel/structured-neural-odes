#!/usr/bin/env python3
"""Data-driven absorption on the tacrolimus regimen benchmark.
For every model, design (sparse / dense) and seed available, the predicted test curves under the TRAINING regimen (each patient's
own arm, doses at 0, 12, 24, 36 h) are rebuilt from the saved run (dose-event, dose-cond: dh_regimen.py; Lu: lu_regimen.py; both
re-executed with the run's own flags, weights loaded, reload checked against the run's JSON) or read from the dumped curves
(EB-VAE, DUMP_CURVES=1), and compared with the noise-free truth on SHAPE:
  C(0)        predicted concentration at t = 0 (patients start drug-free; truth 0), median over patients, ng/mL
  t_peak      time from the last dose (36 h) to the predicted maximum over 36-42 h, median (truth: 1.25 h)
  jump        rise from 35.75 h (just before the last dose) to 36.25 h / rise from 35.75 h to the peak (0 = smooth absorption,
              1 = the whole rise is immediate), median; truth printed alongside. 35.75 h, not 36 h: a dose at t applies before the
              output at t, so a model that jumps already shows the jump at 36 h
  C1, C3      predicted / true concentration 1 h and 3 h after the last dose (the C1 / C3 TDM samples), median ratio
plus the AUC errors of the run (unseen = mean over the 5 unseen regimens of the median |AUC(0-45 h) err|; training regimen).
Figure: per design, one test patient (rule: median true AUC on its training regimen; the sparse and dense runs of a seed have
different test patients, because the training noise draws come first in the data stream), every model, seed 0.
    tacro_shape_compare.py [--seeds 0,1,2,3,4] [--fig-seed 0]   -> results/regimen_dh/tacro/shape_table.txt, figures/shape_models.png"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, runpy, argparse, io, contextlib
import numpy as np, torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', default='0,1,2,3,4'); ap.add_argument('--fig-seed', type=int, default=0)
ap.add_argument('--scen', type=int, default=4, choices=(2, 4))
ap.add_argument('--depot', action='store_true')   # 19/10: the dose-event model of the depot rerun (Bdepot*); own outputs
ap.add_argument('--designs', default='sparse,dense')   # 16/10: scenario 2 (linear) -> results/regimen_dh/tacro_s2/, own outputs
cli = ap.parse_args()
if cli.scen == 2:
    os.environ['TACRO_SCENARIO'] = '2'
ROOT = REPO; os.chdir(ROOT); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts/regimen'))
import gen_tacro_regimens as G
R = 'results/regimen_dh/tacro'; EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen/res_tacro/ctrl'
MODELS = [('Bfixkp4n', 'Dose-event model (ours)'), ('dosecond_n', 'Dose-conditioned field'), ('ebvae', 'EB-VAE'),
          ('luis', 'Lu et al. iScience'), ('lunmi', 'Lu et al. NMI (two-port)')]
if cli.scen == 2:
    assert G.SCENARIO == 2
    R = 'results/regimen_dh/tacro_s2'; EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen/res_tacro_s2/ctrl'
    MODELS = [('Bfixkp3.1n', 'Dose-event model (ours)'), ('ebvae', 'EB-VAE'), ('luis', 'Lu et al. iScience'), ('lunmi', 'Lu et al. NMI (two-port)')]
if cli.depot:
    MODELS = [(('Bdepot3.1n' if cli.scen == 2 else 'Bdepot4n'), 'Dose-event model, depot')] + [m for m in MODELS if not m[0].startswith('B')]
SEEDS = [int(s) for s in cli.seeds.split(',')]
UNSEEN = G.UNSEEN


def quiet_run(path, argv):
    sys.argv = argv
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(path, run_name='shape')


def curves(model, design, seed):
    """(predicted [400, len(EG)] under the training regimen, the run's JSON, truth [400, len(EG)]) or None"""
    tag = f'N100_{design}_s{seed}'
    if model == 'ebvae':
        js, npz = f'{EB}/{tag}.json', f'{EB}/{tag}_curves.npz'
        if not (os.path.exists(js) and os.path.exists(npz)):
            return None
        return np.load(npz)['train'], json.load(open(js)), None
    js, pt = f'{R}/{model}/{tag}.json', f'{R}/{model}/{tag}.pt'
    if not (os.path.exists(js) and os.path.exists(pt)):
        return None
    ck = torch.load(pt, map_location='cpu', weights_only=False); c = ck['cli']; res = json.load(open(js))
    if model.startswith('lu'):
        g = quiet_run('scripts/regimen/lu_regimen.py', ['x', c['model'], '100', design, str(seed), '/tmp/u.json', '--sim', 'tacro'])
        m = g['model']; m.load_state_dict(ck['state_dict']); m.eval()
        times, A = g['regs']['train']
        with torch.no_grad():
            if c['model'] == 'is':   # as published: the encoder reads the regimen predicted (here the training one = factual)
                mu, _ = m.encode(g['rows'](g['y_te'], times, A, cov=g['covs'](g['P_te']))[0])
                pred = m.decode(m.rollout(mu, times, A, g['EG']), g['ctx_te']).numpy() * g['YMAX']
            else:
                theta, x0 = m.encode(g['feats_te'])
                pred = m.simulate(theta, x0, times, A, g['EG']).numpy() * g['YMAX']
    else:
        argv = ['x', '100', design, str(seed), '/tmp/u.json', '--sim', 'tacro', '--enc-dose', c['enc_dose']] + (['--depot', str(c['depot'])] if c.get('depot') else [])
        argv += ['--obsrv-mgl', str(c['obsrv_mgl'])] if c.get('obsrv_mgl') is not None else []
        argv += ['--kick-phys', str(c['kick_phys'])] if c.get('kick_phys') else []
        argv += ['--dose-cond'] if c.get('dose_cond') else []
        g = quiet_run('scripts/regimen/dh_regimen.py', argv)
        g['model'].load_state_dict(ck['state_dict']); g['model'].eval()
        times, A = g['regs']['train']
        with torch.no_grad():
            mu, _ = g['model'].encoder_z0(g['X_te'], g['tp_samp'], static=g['S_te'], run_backwards=True)
            pred = g['decode'](g['simulate'](mu, times, A, g['EG']))[0].numpy() * g['YMAX']
    assert abs(ck['YMAX'] - g['YMAX']) < 1e-9, f'{model} {tag}: data regenerated differently'
    tru = np.stack([np.interp(g['EG'], g['t_eval'], cc) for cc in g['truth']['train']])
    e = 100 * np.median(np.abs(np.trapezoid(pred, g['EG'], axis=1) / np.trapezoid(tru, g['EG'], axis=1) - 1))
    assert abs(e - res['individual']['train']['auc_abs_median']) < 1e-2, f'{model} {tag}: reload gives {e:.3f}, run {res["individual"]["train"]["auc_abs_median"]:.3f}'
    return pred, res, tru


def truth_train(design, seed):
    """the noise-free training-regimen curves of the test patients (data stream of dh_regimen.py)"""
    rng = np.random.default_rng(seed); P_tr = G.draw(200, rng); arm = np.where(np.arange(200) % 2 == 0, G.LOW, G.HIGH)
    S = G.SAMP_BY_DESIGN[design]
    t, c = G.simulate(P_tr, G.TRAIN_TIMES, np.repeat(arm[:, None], 4, 1)); G.noisy(np.stack([np.interp(S, t, ci) for ci in c]), rng)
    P_te = G.draw(400, rng); arm_te = np.where(np.arange(400) % 2 == 0, G.LOW, G.HIGH)
    t, c = G.simulate(P_te, G.TRAIN_TIMES, np.repeat(arm_te[:, None], 4, 1))
    return np.stack([np.interp(G.EG, t, ci) for ci in c]), arm_te, P_te


EG = G.EG
i_pre, i36, i37, i39 = (int(np.argmin(np.abs(EG - t))) for t in (35.75, 36.0, 37.0, 39.0))
w = (EG >= 36.0 - 1e-9) & (EG <= 42.0 + 1e-9)


def shape(c):
    seg = c[:, w]; k = np.argmax(seg, 1)
    t_peak = EG[w][k] - 36.0
    rise = seg.max(1) - c[:, i_pre]
    jump = (c[:, i36 + 1] - c[:, i_pre]) / np.where(rise > 1e-6, rise, np.nan)
    return dict(c0=c[:, 0], t_peak=t_peak, jump=jump)


rows, store = [], {}
for design in cli.designs.split(','):
    for model, lab in MODELS:
        acc = {k: [] for k in ('u', 'o', 'c0', 'tp', 'jump', 'c1', 'c3', 'jump_true', 'tp_true')}
        for s in SEEDS:
            out = curves(model, design, s)
            if out is None:
                continue
            pred, res, tru = out
            if tru is None:
                tru = truth_train(design, s)[0]
            if model == 'ebvae':   # dumped curves: check them against the run's own training-regimen error
                e = 100 * np.median(np.abs(np.trapezoid(pred, EG, axis=1) / np.trapezoid(tru, EG, axis=1) - 1))
                assert abs(e - res['individual']['train']['auc_abs_median']) < 1e-2, f'EB-VAE {design} s{s}: curves give {e:.3f}'
            ind = res['individual']
            acc['u'].append(float(np.mean([ind[g]['auc_abs_median'] for g in UNSEEN]))); acc['o'].append(ind['train']['auc_abs_median'])
            sp, st = shape(pred), shape(tru)
            acc['c0'].append(np.median(sp['c0'])); acc['tp'].append(np.median(sp['t_peak'])); acc['jump'].append(np.nanmedian(sp['jump']))
            acc['tp_true'].append(np.median(st['t_peak'])); acc['jump_true'].append(np.nanmedian(st['jump']))
            acc['c1'].append(np.median(pred[:, i37] / tru[:, i37])); acc['c3'].append(np.median(pred[:, i39] / tru[:, i39]))
            if s == cli.fig_seed:
                store[(model, design)] = (pred, tru)
            print(f'{design} {model} s{s} done', flush=True)
        rows.append((design, lab, acc))

ms = lambda v, f='.1f': '-' if not v else (f'{np.mean(v):{f}}' + (f' +- {np.std(v, ddof=1):{f}}' if len(v) > 1 else ''))
L = [__doc__.strip().split('\n')[0], '', 'shape on the training regimen (median over 400 test patients, then mean +- sd over seeds); truth: C(0) 0, '
     't_peak / jump below', '']
hdr = f'{"design":7s} {"model":28s} {"n":>2s} {"unseen AUC":>13s} {"train AUC":>13s} {"C(0) ng/mL":>13s} {"t_peak h":>12s} {"jump":>12s} {"C1 ratio":>12s} {"C3 ratio":>12s}'
L.append(hdr)
for design, lab, a in rows:
    L.append(f'{design:7s} {lab:28s} {len(a["u"]):2d} {ms(a["u"]):>13s} {ms(a["o"]):>13s} {ms(a["c0"]):>13s} {ms(a["tp"], ".2f"):>12s} '
             f'{ms(a["jump"], ".2f"):>12s} {ms(a["c1"], ".2f"):>12s} {ms(a["c3"], ".2f"):>12s}')
    if a['tp_true'] and lab.startswith('Dose-event'):
        L.append(f'{design:7s} {"(truth)":28s}    {"":>13s} {"":>13s} {"0":>13s} {ms(a["tp_true"], ".2f"):>12s} {ms(a["jump_true"], ".2f"):>12s} '
                 f'{"1":>12s} {"1":>12s}')
open(f'{R}/shape_table' + ('_depot' if cli.depot else '') + ('' if cli.designs == 'sparse,dense' else '_' + cli.designs.replace(',', '_')) + '.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))

# ---- figure: one patient, every model, sparse vs dense
if store and not cli.depot and cli.designs == 'sparse,dense':   # 19/10: the figure only for the original runs
    pick = {}
    for design in ('sparse', 'dense'):
        tru0, arm0, _ = truth_train(design, cli.fig_seed)
        auc = np.trapezoid(tru0, EG, axis=1); pick[design] = (int(np.argsort(auc)[len(auc) // 2]), arm0)
    SURF, INK, INK2, GRID, C_MOD = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3de', '#2a78d6'
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8.5, 'axes.facecolor': SURF, 'figure.facecolor': SURF,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.titlesize': 9, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})
    fig, axes = plt.subplots(len(MODELS), 2, figsize=(8.6, 1.75 * len(MODELS) + 0.8), sharex=True, sharey=True)
    for r, (model, lab) in enumerate(MODELS):
        for k, design in enumerate(('sparse', 'dense')):
            ax = axes[r, k]; i, arm0 = pick[design]
            S = G.SAMP_DENSE if design == 'dense' else G.SAMP_SPARSE
            if (model, design) not in store:
                ax.text(0.5, 0.5, 'run pending', transform=ax.transAxes, ha='center', color=INK2); continue
            pred, tru = store[(model, design)]
            ax.plot(EG, tru[i], color=INK, lw=1.3)
            ax.plot(EG, pred[i], color=C_MOD, lw=1.5, ls=(0, (4, 1.6)))
            ax.plot(S, np.interp(S, EG, tru[i]), '|', color='#eb6834', ms=7, mew=1.4)
            for t in G.TRAIN_TIMES:
                ax.plot([t], [0], marker='^', color=INK2, ms=4, clip_on=False)
            ax.grid(axis='y', color=GRID, lw=0.6); ax.set_xticks([0, 12, 24, 36, 45]); ax.set_xlim(0, G.T_END)
            if r == 0:
                ax.set_title(f'{design} sampling ({len(S)} samples, orange ticks)\ntest patient {i}, {arm0[i]:g} mg arm')
            if k == 0:
                ax.set_ylabel(lab.replace(' (', '\n('), fontsize=8.3)
    axes[-1, 0].set_xlabel('hours'); axes[-1, 1].set_xlabel('hours')
    fig.suptitle(f'Median-AUC test patient per design (same for every model), training regimen, seed {cli.fig_seed}\n'
                 'black = truth, blue = prediction', x=0.01, ha='left', fontsize=9.3, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(f'{R}/figures', exist_ok=True); fig.savefig(f'{R}/figures/shape_models.png', dpi=170)
    print(f'wrote {R}/figures/shape_models.png (test patients {pick["sparse"][0]} sparse, {pick["dense"][0]} dense)')
