#!/usr/bin/env python3
"""Figures for the Lu comparison on Baaz's regimen test.
  test1_identity.{png,pdf}: (A) latent map of the BEST-TUNED Lu iScience (lr 1e-4, wd 1e-3, sparse N100 seed 0): 400 test
      patients encoded with the training regimen (grey), and 3 patients (log ke quantiles 15/50/85 %, not hand-picked) re-encoded
      with the SAME observations but each target regimen's dosing channels; (B) identity ratio per target regimen, every run;
      (C) log ke probe fitted on z0(train regimen), applied to z0(target regimen).
  test2_titration.{png,pdf}: (A) training-arm assignment, randomised vs titrated (N100 sparse seed 0); (B) dose elasticity
      350 -> 850 mg per model, randomised vs titrated, true value; (C) unseen-regimen |AUC err| per model.
    python scripts/regimen/plot_tests.py"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, glob, numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, f'{REPO}/third_party/baaz_ebvae/ours/regimen')
import gen_regimens as G
R = 'results/regimen_dh'; EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'; OUT = f'{R}/figures'
os.makedirs(OUT, exist_ok=True)
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3de'
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'       # validated categorical slots 1-4 (light)
REGS = ('train', 'amt350', 'amt600', 'amt850', 'time400', 'time800'); UNSEEN = REGS[1:]
RLAB = {'train': 'training regimen', 'amt350': '350 mg x4', 'amt600': '600 mg x4', 'amt850': '850 mg x4',
        'time400': '400 mg at 0/3/8 h', 'time800': '800 mg at 0/3/8 h'}
MK = {'train': 'o', 'amt350': 'v', 'amt600': 's', 'amt850': '^', 'time400': 'D', 'time800': 'P'}
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
                     'axes.titlecolor': INK, 'axes.titlelocation': 'left'})


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def grid(ax, axis='y'):
    ax.grid(axis=axis, color=GRID, lw=0.8); ax.set_axisbelow(True)


# ============================================================== Test 1
def encodings(ck, design='sparse', seed=0):
    """z0 of the 400 test patients encoded with each regimen's dosing channels (same observations)"""
    sys.argv = ['lu_regimen.py', 'is', '100', design, str(seed), '/tmp/_x.json', '--iters', '0', '--no-check']
    src = open('scripts/regimen/lu_regimen.py').read().replace("if __name__ == '__main__':\n    main()", '')
    g = {'__name__': 'plot'}; exec(compile(src, 'lu_regimen.py', 'exec'), g)
    m = g['model']; m.load_state_dict(torch.load(ck, weights_only=False)['state_dict']); m.eval()
    with torch.no_grad():
        z = {n: m.encode(g['rows'](g['y_te'], *g['regs'][n])[0])[0].numpy() for n in REGS}
    return z, np.log(g['P_te'][:, 1])


SHORT = {'train': 'training\nregimen', 'amt350': '350 mg\nx4', 'amt600': '600 mg\nx4', 'amt850': '850 mg\nx4',
         'time400': '400 mg\n0/3/8 h', 'time800': '800 mg\n0/3/8 h'}


def fig_test1():
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import r2_score
    ck = f'{R}/luis_sweep/lr3e-4_wd0/N100_sparse_s0.pt'   # tuned AND still individualised (factual err 8 %, ke R² 0.87)
    z, lke = encodings(ck)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw=dict(width_ratios=[1, 1.05, 1.05]))
    # (A) readout of log ke from the latent: fitted on the training-regimen encodings, applied to target-regimen ones
    ax = axes[0]
    probe = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(z['train'], lke)
    pts = [('train', cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 13)), z['train'], lke, cv=5), '#a9a8a2', 'o'),
           ('amt350', probe.predict(z['amt350']), C1, 'v'), ('time400', probe.predict(z['time400']), C2, 'D')]
    lo, hi = lke.min() - 0.2, lke.max() + 0.2
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls='--')
    for name, pred, col, mk in pts:
        ax.scatter(lke, pred, s=12, marker=mk, color=col, alpha=0.7, lw=0,
                   label=f'{RLAB[name]}  (R² {r2_score(lke, pred):.2f})')
    ax.set_xlim(lo, hi); ax.set_ylim(min(lo, min(p[1].min() for p in pts) - 0.1), max(hi, max(p[1].max() for p in pts) + 0.1))
    ax.set_xlabel('true log ke of the test patient'); ax.set_ylabel('log ke read from the patient\'s latent')
    ax.legend(fontsize=7.5, frameon=False, loc='upper left', title='encoder fed the dosing of', title_fontsize=7.5)
    ax.set_title('A. Same patients, same data, other target regimen'); grid(ax, 'both')
    ax.text(0.0, -0.22, 'Lu iScience, tuned (lr 3e-4, no weight decay), sparse, N = 100/arm, seed 0; readout = ridge fitted on\n'
            'the training-regimen latents (cross-validated there), applied unchanged. The lower-error tuned settings encode\n'
            'no ke at all (R² 0.26 / 0.00 on the training regimen): they behave as population models.', transform=ax.transAxes, fontsize=7, color=INK2)
    # (B) identity ratio per regimen, every run
    ax = axes[1]
    runs = [('default settings, sparse (3 seeds)', [f'{R}/luis/N100_sparse_s{s}.json' for s in (0, 1, 2)], C2, 'o', 30),
            ('default settings, dense (1 seed)', [f'{R}/luis/N100_dense_s0.json'], C2, 'D', 30),
            ('tuned settings, sparse seed 0', [p for p in sorted(glob.glob(f'{R}/luis_sweep/*/N100_sparse_s0.json')) if '_wd0.1' not in p], C1, 'o', 14)]
    x = np.arange(len(UNSEEN))
    for k, (lab, files, col, mk, sz) in enumerate(runs):
        js = [j for f in files for j in [load(f)] if j is not None and 'identity' in j]
        for j in js:
            ax.scatter(x + (k - 1) * 0.2, [j['identity'][g] for g in UNSEEN], color=col, marker=mk, s=sz, alpha=0.85, lw=0,
                       label=lab if j is js[0] else None)
    ax.axhline(1, color=INK2, lw=1, ls='--')
    ax.text(len(UNSEEN) - 0.55, 1.04, 'as far apart as two random patients', fontsize=7.5, color=INK2, va='bottom', ha='right')
    ax.axhline(0, color=C3, lw=2.5)
    ax.text(-0.45, 0.05, 'B, B2, Lu NMI, EB-VAE: 0 by construction', fontsize=7.5, color=INK, va='bottom')
    ax.set_xticks(x, [SHORT[g] for g in UNSEEN], fontsize=7.5); ax.set_ylim(-0.1, None)
    ax.set_ylabel('identity ratio  E|z0(r) - z0(train)|² / 2 tr Cov z0'); grid(ax)
    ax.set_title('B. How far the same patient moves'); ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    ax.set_xlabel('target regimen fed to the encoder')
    # (C) readout R² per regimen, every run
    ax = axes[2]
    for k, (lab, files, col, mk, sz) in enumerate(runs):
        js = [j for f in files for j in [load(f)] if j is not None and 'identity_probe' in j]
        for j in js:
            pr = j['identity_probe']['log ke']
            ax.plot(np.arange(len(REGS)) + (k - 1) * 0.12, np.clip([pr['cv_r2_train']] + [pr[g] for g in UNSEEN], -2.5, 1),
                    color=col, marker=mk, ms=4 if sz > 20 else 3, lw=0.8, alpha=0.7, label=lab if j is js[0] else None)
    ax.axhline(0, color=INK2, lw=1, ls='--')
    ax.text(-0.3, -0.12, 'below 0: worse than the population mean', fontsize=7.5, color=INK2, va='top')
    ax.set_xticks(np.arange(len(REGS)), [SHORT[g] for g in REGS], fontsize=7.5)
    ax.set_ylim(-2.6, 1.08); ax.set_ylabel('R² of the log ke readout (clipped at -2.5)'); grid(ax)
    ax.set_xlabel('target regimen fed to the encoder')
    ax.set_title('C. Elimination rate read from the latent')
    fig.tight_layout(); fig.subplots_adjust(bottom=0.24, wspace=0.28)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/test1_identity.{ext}', dpi=200)
    print('test1 ->', f'{OUT}/test1_identity.png')


# ============================================================== Test 2
EG = np.arange(0, 30.0001, 0.25); _truth = {}


def true_auc(d, s):
    if (d, s) not in _truth:
        SAMP = np.arange(0, 30.25, 0.5) if d == 'dense' else np.array([0, 2, 7.5, 15.5, 23.5, 28.0])
        rng = np.random.default_rng(s); P_tr = G.draw(200, rng)
        A_tr = np.repeat(np.where(np.arange(200) % 2 == 0, 400.0, 800.0)[:, None], 4, 1)
        te, c = G.simulate(P_tr, [0, 8, 16, 24], A_tr); G.noisy(np.stack([np.interp(SAMP, te, ci) for ci in c]), rng)
        P_te = G.draw(400, rng); out = {}
        for name in ('amt350', 'amt850'):
            times, amts = G.REGIMENS[name]; te, c = G.simulate(P_te, times, np.tile(np.array(amts, float), (400, 1)))
            out[name] = np.trapezoid(np.stack([np.interp(EG, te, ci) for ci in c]), EG, axis=1)
        _truth[(d, s)] = out
    return _truth[(d, s)]


def run(model, cond, d, s):
    tag = f'N100_{d}_s{s}'
    if model == 'ebvae':
        return load(f'{EB}/res_titr/{"ctrl" if cond == "ctrl" else "titr0.8"}/{tag}.json')
    return load(f'{R}/{model}/{tag}.json' if cond == 'ctrl' else f'{R}/titr0.8/{model}/{tag}.json')


def elasticity(j, d, s):
    a = {}
    for n in ('amt350', 'amt850'):
        ind = j['individual'][n]
        a[n] = np.array(ind['auc_pred_per_patient']) if 'auc_pred_per_patient' in ind else (1 + np.array(ind['auc_err_per_patient'])) * true_auc(d, s)[n]
    t = true_auc(d, s); k = np.log(850 / 350)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.log(a['amt850'] / a['amt350']) / k, np.log(t['amt850'] / t['amt350']) / k


MODELS = (('B2', 'B2 (ours)', C1), ('luis', 'Lu iScience', C2), ('ebvae', 'EB-VAE (Baaz)', C3), ('B', 'B (ours)', C4))
DESIGNS = (('sparse', (0, 1, 2), 'sparse (3 seeds)'), ('dense', (0,), 'dense (1 seed)'))


def fig_test2():
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0), gridspec_kw=dict(width_ratios=[0.75, 1.1, 1.1]))
    # (A) arm assignment
    ax = axes[0]
    P = G.draw(200, np.random.default_rng(1)); ke = P[:, 1]            # seed 1 = a typical draw (see caption for seed 0)
    arms = {'randomised': np.where(np.arange(200) % 2 == 0, 400.0, 800.0), 'titrated': G.titrate(P, 0.8, 1)[0]}
    rng = np.random.default_rng(1)
    for k, (cond, arm) in enumerate(arms.items()):
        for a, col, dx in ((400, INK2, -0.17), (800, C2, 0.17)):
            v = ke[arm == a]; xs = k + dx + rng.uniform(-0.07, 0.07, len(v))
            ax.scatter(xs, v, s=8, color=col, alpha=0.6, lw=0)
            ax.plot([k + dx - 0.12, k + dx + 0.12], [np.median(v)] * 2, color=INK, lw=2)
        r = np.corrcoef(np.log(ke), arm == 800)[0, 1]
        ax.text(k, ke.min() * 0.62, f'corr(log ke, 800 mg)\n{r:+.2f}', ha='center', va='top', fontsize=7.5, color=INK)
    ax.set_yscale('log'); ax.set_ylim(ke.min() * 0.3, ke.max() * 1.5); ax.set_xticks([0, 1], ['randomised', 'titrated'])
    ax.set_ylabel('true elimination rate ke (1/h, log scale)'); grid(ax)
    ax.legend(handles=[Line2D([], [], ls='', marker='o', color=INK2, label='400 mg arm'), Line2D([], [], ls='', marker='o', color=C2, label='800 mg arm'),
                       Line2D([], [], color=INK, lw=2, label='median')], fontsize=7.5, frameon=False, loc='upper left', ncol=3,
              bbox_to_anchor=(0, 1.0), handletextpad=0.3, columnspacing=0.8)
    ax.set_title('A. Who gets 800 mg in training')
    ax.text(0.0, -0.16, 'titrated: fast eliminators on 800 mg. Training patients, N = 100/arm,\nseed 1; test patients keep randomised arms. Seed 0\'s randomised arms\n'
            'happen to correlate +0.27 (most extreme of 500 draws; seeds 1, 2: -0.13, -0.11).', transform=ax.transAxes, fontsize=7, color=INK2, va='top')
    # (B) elasticity
    ax = axes[1]
    y = 0; ticks = []; truth_e = []
    for d, seeds, dlab in DESIGNS:
        for m, lab, col in MODELS:
            meds = {}
            for cond in ('ctrl', 'titr'):
                es = [elasticity(j, d, s) for s in seeds for j in [run(m, cond, d, s)] if j is not None]
                if es:
                    ep = np.concatenate([e[0] for e in es]); ok = np.isfinite(ep); meds[cond] = np.median(ep[ok])
                    truth_e.append(np.median(np.concatenate([e[1] for e in es])))
            if len(meds) == 2:
                ax.plot([meds['ctrl'], meds['titr']], [y, y], color=col, lw=2, zorder=1)
                ax.scatter(meds['ctrl'], y, s=60, facecolor=SURF, edgecolor=col, lw=2, zorder=2)
                ax.scatter(meds['titr'], y, s=60, color=col, edgecolor=SURF, lw=1, zorder=3)
                ax.text(1.22, y, f'{meds["ctrl"]:.2f} -> {meds["titr"]:.2f}', va='center', fontsize=7.5, color=INK)
            ticks.append((y, f'{lab}, {dlab}')); y -= 1
        y -= 0.6
    te = np.median(truth_e); ax.axvline(te, color=INK, lw=1, ls='--')
    ax.text(te - 0.02, y + 0.35, f'truth {te:.2f}', ha='right', va='center', fontsize=7.5, color=INK)
    ax.text(1.22, 0.75, 'random -> titrated', fontsize=7.5, color=INK2, va='center')
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=7.5); ax.set_xlim(-0.1, 1.6); ax.set_ylim(y + 0.2, 1.1)
    ax.set_xlabel('dose elasticity  d log AUC / d log dose\n(350 -> 850 mg, median test patient)'); grid(ax, 'x')
    ax.legend(handles=[Line2D([], [], ls='', marker='o', mfc=SURF, mec=INK2, mew=2, label='randomised training'),
                       Line2D([], [], ls='', marker='o', color=INK2, label='titrated training')], fontsize=7.5, frameon=False, loc='upper left',
              bbox_to_anchor=(0, -0.2), ncol=2)
    ax.set_title('B. Does AUC still scale with dose?')
    # (C) unseen-regimen error
    ax = axes[2]
    y = 0; ticks = []
    for d, seeds, dlab in DESIGNS:
        for m, lab, col in MODELS:
            vals = {cond: [float(np.mean([j['individual'][g]['auc_abs_median'] for g in UNSEEN])) for s in seeds for j in [run(m, cond, d, s)] if j is not None]
                    for cond in ('ctrl', 'titr')}
            if vals['ctrl'] and vals['titr']:
                a, b = np.mean(vals['ctrl']), np.mean(vals['titr'])
                ax.plot([a, b], [y, y], color=col, lw=2, zorder=1)
                ax.scatter(vals['ctrl'], [y + 0.18] * len(vals['ctrl']), s=10, facecolor=SURF, edgecolor=col, lw=1)
                ax.scatter(vals['titr'], [y - 0.18] * len(vals['titr']), s=10, color=col, lw=0)
                ax.scatter(a, y, s=60, facecolor=SURF, edgecolor=col, lw=2, zorder=2); ax.scatter(b, y, s=60, color=col, edgecolor=SURF, lw=1, zorder=3)
                ax.text(108, y, f'{a:.0f} -> {b:.0f}', va='center', fontsize=7.5, color=INK)
            ticks.append((y, f'{lab}, {dlab}')); y -= 1
        y -= 0.6
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=7.5); ax.set_xlim(0, 125); ax.set_ylim(y + 0.2, 1.1)
    ax.text(108, 0.75, 'random -> titrated', fontsize=7.5, color=INK2, va='center')
    ax.set_xlabel('median |AUC(0-30 h) error| %\n(mean of the 5 unseen regimens; small dots = seeds)'); grid(ax, 'x')
    ax.set_title('C. Error on unseen regimens')
    fig.text(0.01, 0.01, 'Lu iScience = published design with default (untuned) optimiser settings; Lu NMI omitted (did not learn a dose response). '
             'Truth: Baaz et al. linear 1-compartment simulator.', fontsize=7.5, color=INK2)
    fig.tight_layout(); fig.subplots_adjust(bottom=0.24, wspace=0.75)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/test2_titration.{ext}', dpi=200)
    print('test2 ->', f'{OUT}/test2_titration.png')


if __name__ == '__main__':
    fig_test1(); fig_test2()
