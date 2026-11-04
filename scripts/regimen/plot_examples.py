#!/usr/bin/env python3
"""Patient-level figures for the Lu comparison on Baaz's regimen test: what goes in, what comes out,
what is trained on and what is scored. Models are reloaded from their saved checkpoints (B, B2, Lu iScience) and evaluated on
the exact data of their run (each script re-executed with its own flags, YMAX checked against the checkpoint); EB-VAE curves
come from deterministic reruns with DUMP_CURVES=1 (scripts/regimen/queue_ebvae_curves.sh), shown when available.
  ex1_setup.{png,pdf}   N100 sparse seed 1: training patients (doses + 6 noisy samples), one test patient's input, its six
                        target curves (never observed; scored on AUC 0-30 h).
  ex2_test1.{png,pdf}   Lu iScience tuned & individualised (lr 3e-4, wd 0; sparse seed 0): two test patients, predictions when
                        the encoder is fed the TARGET regimen (as published) vs the OBSERVED regimen, and the log ke read from
                        the latent for each target regimen.
  ex3_test2.{png,pdf}   N100 sparse seed 1: one test patient, every model trained on randomised vs titrated arms, predicted
                        under 350 / 850 mg x4, and AUC vs dose.
Patients are chosen by rule (test patients observed on 400 mg, true ke quantiles), not by eye.
    python scripts/regimen/plot_examples.py"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, numpy as np, torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R = 'results/regimen_dh'; EBC = f'{REPO}/third_party/baaz_ebvae/ours/regimen/res_curves'; OUT = f'{R}/figures'
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3de'
C1, C2, C3, C4 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
RLAB = {'train': 'observed regimen', 'amt350': '350 mg x4 (0/8/16/24 h)', 'amt600': '600 mg x4', 'amt850': '850 mg x4 (0/8/16/24 h)',
        'time400': '400 mg at 0/3/8 h', 'time800': '800 mg at 0/3/8 h'}
REGS = ('train', 'amt350', 'amt600', 'amt850', 'time400', 'time800')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9.5, 'axes.titleweight': 'bold',
                     'axes.titlecolor': INK, 'axes.titlelocation': 'left'})


def run_script(path, argv):
    sys.argv = [path] + argv
    src = open(path).read().replace("if __name__ == '__main__':\n    main()", '')
    g = {'__name__': 'examples'}; exec(compile(src, path, 'exec'), g)
    return g


def doses(ax, times, amounts, ymax, color=INK2, labels=None):
    for k, (t, a) in enumerate(zip(times, amounts)):
        ax.annotate('', xy=(t, 0), xytext=(t, -0.09 * ymax), arrowprops=dict(arrowstyle='-|>', color=color, lw=1.2), annotation_clip=False)
        ax.text(t + 0.4, -0.085 * ymax, labels[k] if labels else f'{a:.0f}', fontsize=6.5, color=color, va='bottom')


def style(ax, ymax, xlabel=True):
    ax.set_xlim(-0.5, 30.5); ax.set_ylim(-0.1 * ymax, ymax); ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel('time (h)')


# ---------------------------------------------------------------- model loaders (predicted curves on EG for chosen patients)
def lu_is(seed, titr, ck, idx, names):
    g = run_script('scripts/regimen/lu_regimen.py', ['is', '100', 'sparse', str(seed), '/tmp/_x.json', '--iters', '0', '--no-check']
                   + (['--titrate', '0.8'] if titr else []))
    st = torch.load(ck, weights_only=False); assert abs(st['YMAX'] - g['YMAX']) < 1e-6, 'data of this run not reproduced'
    m = g['model']; m.load_state_dict(st['state_dict']); m.eval()
    out = {}
    with torch.no_grad():
        for n in names:
            times, A = g['regs'][n]
            for key, f in (('target', g['rows'](g['y_te'][idx], times, A[idx])[0]),
                           ('factual', g['rows'](g['y_te'][idx], g['TRAIN_TIMES'], g['A_te'][idx])[0])):
                out[(n, key)] = m.decode(m.rollout(m.encode(f)[0], times, A[idx], g['EG']), g['ctx_te'][idx]).numpy() * g['YMAX']
    return out, g, m


def dh(seed, titr, ck, idx, names, v2=False):
    g = run_script('scripts/regimen/dh_regimen_v2.py' if v2 else 'scripts/regimen/dh_regimen.py',
                   ['100', 'sparse', str(seed), '/tmp/_x.json', '--iters', '0', '--no-check'] + (['--titrate', '0.8'] if titr else []))
    st = torch.load(ck, weights_only=False); assert abs(st['YMAX'] - g['YMAX']) < 1e-6, 'data of this run not reproduced'
    out = {}
    with torch.no_grad():
        if v2:
            g['encoder'].load_state_dict(st['encoder']); g['osplit'].load_state_dict(st['osplit']); g['lin'].load_state_dict(st['lin'])
            g['encoder'].eval(); enc = g['encoder']
            pred = lambda mu, t, A: g['predict'](mu, t, A, g['EG'])[0].numpy() * g['YMAX']
        else:
            g['model'].load_state_dict(st['state_dict']); g['model'].eval(); enc = g['model'].encoder_z0
            pred = lambda mu, t, A: g['decode'](g['simulate'](mu, t, A, g['EG']))[0].numpy() * g['YMAX']
        mu, _ = enc(g['X_te'][idx], g['tp_samp'], static=g['S_te'][idx], run_backwards=True)
        for n in names:
            times, A = g['regs'][n]; out[n] = pred(mu, times, A[idx])
    return out, g


def ebvae(titr, idx, names):
    p = f'{EBC}/N100_sparse_s1_{"0.8" if titr else "none"}_curves.npz'
    if not os.path.exists(p):
        return None
    d = np.load(p); return {n: d[n][idx] for n in names}


def truth_curve(g, name, idx):
    return g['t_eval'], g['truth'][name][idx]


# ================================================================ figure 1: what the models see
def fig_setup():
    g = run_script('scripts/regimen/lu_regimen.py', ['is', '100', 'sparse', '1', '/tmp/_x.json', '--iters', '0', '--no-check'])
    G, SAMP = g['G'], g['SAMP']
    fig = plt.figure(figsize=(15, 8.2)); gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1], hspace=0.6, wspace=0.45)
    # (A) training data: 3 patients per arm (ke quantiles 20/50/80 % within the arm)
    ax = fig.add_subplot(gs[0, :4]); P, A_tr, y_tr = g['P_tr'], g['A_tr'], g['y_tr']
    te, ctr = G.simulate(P, g['TRAIN_TIMES'], A_tr)
    ymax = 1.08 * y_tr.max()
    for arm, col in ((400, INK2), (800, C2)):
        ids = np.where(A_tr[:, 0] == arm)[0]; ke = P[ids, 1]
        for q in (0.2, 0.5, 0.8):
            i = ids[int(np.argmin(np.abs(ke - np.quantile(ke, q))))]
            ax.plot(te, ctr[i], color=col, lw=0.9, ls=':', alpha=0.8)
            ax.plot(SAMP, y_tr[i], color=col, marker='o', ms=5, lw=0, mec=SURF, mew=0.8)
    doses(ax, g['TRAIN_TIMES'], [400] * 4, ymax, labels=['400 or 800 mg'] * 4)
    style(ax, ymax); ax.set_ylabel('concentration (mg/L)')
    ax.set_title('A. TRAINING DATA: 200 patients, 400 mg (grey) or 800 mg (orange) x4, 6 noisy samples each')
    ax.legend(handles=[Line2D([], [], color=INK2, marker='o', lw=0, label='observed samples (what the model trains on)'),
                       Line2D([], [], color=INK2, ls=':', label='true curve (never seen)')], fontsize=7.5, frameon=False, loc='upper right')
    # (B) the test patient's input
    i400 = np.where(g['A_te'][:, 0] == 400)[0]; ke = g['P_te'][i400, 1]; p = int(i400[int(np.argmin(np.abs(ke - np.median(ke))))])
    ax = fig.add_subplot(gs[0, 4:]); ymax_t = 1.15 * max(g['truth'][n][p].max() for n in REGS)
    ax.plot(SAMP, g['y_te'][p], color=INK, marker='o', ms=6, lw=0)
    doses(ax, g['TRAIN_TIMES'], g['A_te'][p], ymax_t); style(ax, ymax_t)
    ax.set_title(f'B. TEST INPUT: a new patient\n(ke {g["P_te"][p, 1]:.2f} /h) on 400 mg x4, 6 samples')
    ax.text(0.02, 0.97, 'the encoder sees only these points\n(+ the doses given)', transform=ax.transAxes, fontsize=7.5, color=INK2, va='top')
    # (C) targets: the same patient under every regimen
    for k, n in enumerate(REGS):
        ax = fig.add_subplot(gs[1, k]); times, A = g['regs'][n]
        ax.plot(g['t_eval'], g['truth'][n][p], color=INK, lw=1.5)
        ax.fill_between(g['t_eval'], 0, g['truth'][n][p], color=C1, alpha=0.12, lw=0)
        doses(ax, times, A[p], ymax_t); style(ax, ymax_t)
        ax.set_title(('C. TEST TARGETS (never observed)\n' if k == 0 else '\n') + RLAB[n].replace(' (0/8/16/24 h)', ''), fontsize=8.5)
        if k == 0:
            ax.set_ylabel('true concentration (mg/L)')
        if n != 'train':
            ax.text(0.97, 0.97, 'unseen\nregimen', transform=ax.transAxes, fontsize=7, color=C2, ha='right', va='top')
    fig.text(0.01, 0.035, 'Training: every model reconstructs the observed samples (Lu iScience / NMI: MSE; B / B2 / EB-VAE: likelihood + prior). '
             'Titrated variant (test 2): same patients, but the fast eliminators get the 800 mg arm.', fontsize=7.5, color=INK2)
    fig.text(0.01, 0.01, 'Scored: the AUC over 0-30 h (shaded) predicted for this patient under each regimen vs the true AUC. Baaz et al. '
             'simulator (linear 1-compartment, the patient starts with drug on board); N100/arm, sparse sampling, seed 1; patient = '
             'median ke among test patients on 400 mg.', fontsize=7.5, color=INK2)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/ex1_setup.{ext}', dpi=200, bbox_inches='tight')
    print('ex1 ->', f'{OUT}/ex1_setup.png')


# ================================================================ figure 2: test 1 on two patients
def fig_test1():
    from sklearn.linear_model import RidgeCV
    ck = f'{R}/luis_sweep/lr3e-4_wd0/N100_sparse_s0.pt'
    g0 = run_script('scripts/regimen/lu_regimen.py', ['is', '100', 'sparse', '0', '/tmp/_x.json', '--iters', '0', '--no-check'])
    i400 = np.where(g0['A_te'][:, 0] == 400)[0]; ke = g0['P_te'][i400, 1]
    pats = [int(i400[int(np.argmin(np.abs(ke - np.quantile(ke, q))))]) for q in (0.2, 0.8)]
    names = ('train', 'amt350', 'time400')
    out, g, m = lu_is(0, False, ck, np.array(pats), names)
    with torch.no_grad():   # log ke readout: ridge on the training-regimen latents of all 400 test patients
        z = {n: m.encode(g['rows'](g['y_te'], *g['regs'][n])[0])[0].numpy() for n in REGS}
    probe = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(z['train'], np.log(g['P_te'][:, 1]))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.85]))
    for r, p in enumerate(pats):
        ymax = 1.2 * max(g['truth'][n][p].max() for n in names)
        for c, n in enumerate(names):
            ax = axes[r, c]; times, A = g['regs'][n]
            ax.plot(g['t_eval'], g['truth'][n][p], color=INK, lw=1.6, label='truth')
            if n == 'train':
                ax.plot(g['SAMP'], g['y_te'][p], color=INK, marker='o', ms=6, lw=0, label='observed samples = input')
                ax.plot(g['EG'], out[(n, 'target')][r], color=C2, lw=1.6, label='Lu iScience fit')
            else:
                ax.plot(g['EG'], out[(n, 'target')][r], color=C2, lw=1.8, label='as published: encoder fed this target regimen')
                ax.plot(g['EG'], out[(n, 'factual')][r], color=C1, lw=1.4, ls='--', label='encoder fed the observed regimen')
            doses(ax, times, A[p], ymax); style(ax, ymax, xlabel=(r == 1))
            if c == 0:
                ax.set_ylabel(f'patient {"1" if r == 0 else "2"}: ke {g["P_te"][p, 1]:.2f} /h\nconcentration (mg/L)')
            if r == 0:
                ax.set_title({'train': 'INPUT: observed on 400 mg x4\n(and the fit to it)', 'amt350': 'PREDICTED: 350 mg x4',
                              'time400': 'PREDICTED: 400 mg at 0/3/8 h'}[n])
            if r == 0 and c in (0, 1):
                ax.legend(fontsize=7, frameon=False, loc='upper left')
        # log ke read from this patient's latent under each target regimen
        ax = axes[r, 3]
        reads = [float(probe.predict(z[n][p:p + 1])[0]) for n in REGS]
        ax.axhline(np.log(g['P_te'][p, 1]), color=INK, lw=1.4)
        ax.text(5.4, np.log(g['P_te'][p, 1]) + 0.04, 'true log ke', ha='right', va='bottom', fontsize=7.5, color=INK)
        ax.scatter(range(len(REGS)), reads, color=[INK2] + [C2] * 5, s=40, zorder=3)
        ax.set_xticks(range(len(REGS)), ['observed', '350\nx4', '600\nx4', '850\nx4', '400\n0/3/8', '800\n0/3/8'], fontsize=7.5)
        ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True); ax.set_ylabel('log ke read from the latent')
        if r == 0:
            ax.set_title('WHO THE MODEL THINKS\nthe patient is, per target regimen')
        if r == 1:
            ax.set_xlabel('target regimen fed to the encoder')
    fig.text(0.01, -0.035, 'Lu iScience, tuned and individualised (lr 3e-4, no weight decay), sparse N100/arm seed 0. Same patient, same 6 '
             'observations in every column; only the dosing channels given to the encoder change. Patients: ke quantiles 20 % and 80 % '
             'among test patients on 400 mg. Readout = ridge fitted on the observed-regimen latents of all 400 test patients.\n'
             'Mechanism: in training the dosing channels always match the observed concentrations, so the encoder answers "which patient '
             'produced these concentrations under THIS dosing?". Fed a higher counterfactual dose with the same concentrations, it concludes '
             'the patient eliminates faster (and slower for a lower dose), cancelling most of the dose change. Same direction in every seed '
             '(median shift of read log ke at 850 mg +0.19 to +0.34; at 350 mg -0.10 to -0.32).', fontsize=7.5, color=INK2, wrap=True)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/ex2_test1.{ext}', dpi=200, bbox_inches='tight')
    print('ex2 ->', f'{OUT}/ex2_test1.png')


# ================================================================ figure 3: test 2 on one patient
def fig_test2():
    names = ('amt350', 'amt600', 'amt850')
    g = run_script('scripts/regimen/lu_regimen.py', ['is', '100', 'sparse', '1', '/tmp/_x.json', '--iters', '0', '--no-check'])
    i400 = np.where(g['A_te'][:, 0] == 400)[0]; ke = g['P_te'][i400, 1]; p = int(i400[int(np.argmin(np.abs(ke - np.median(ke))))])
    idx = np.array([p]); EG = g['EG']
    curves = {}
    for titr in (False, True):
        sub = 'titr0.8/' if titr else ''
        curves[('B2', titr)] = {n: v[0] for n, v in dh(1, titr, f'{R}/{sub}B2/N100_sparse_s1.pt', idx, names, v2=True)[0].items()}
        curves[('B', titr)] = {n: v[0] for n, v in dh(1, titr, f'{R}/{sub}B/N100_sparse_s1.pt', idx, names)[0].items()}
        o = lu_is(1, titr, f'{R}/{sub}luis/N100_sparse_s1.pt', idx, names)[0]
        curves[('luis', titr)] = {n: o[(n, 'target')][0] for n in names}
        e = ebvae(titr, idx, names)
        curves[('ebvae', titr)] = {n: v[0] for n, v in e.items()} if e else None
    MODELS = (('B2', 'B2 (ours)', C1), ('ebvae', 'EB-VAE (Baaz)', C3), ('B', 'B (ours)', C4), ('luis', 'Lu iScience', C2))
    tr = {n: g['truth'][n][p] for n in names}; auc_t = [np.trapezoid(np.interp(EG, g['t_eval'], tr[n]), EG) for n in names]
    ymax = 1.25 * max(tr['amt850'].max(), max(c['amt850'].max() for c in curves.values() if c))
    fig, axes = plt.subplots(len(MODELS), 3, figsize=(13, 11), gridspec_kw=dict(width_ratios=[1, 1, 0.75]))
    for r, (m, lab, col) in enumerate(MODELS):
        for c, n in enumerate(('amt350', 'amt850')):
            ax = axes[r, c]; times, A = g['regs'][n]
            ax.plot(g['t_eval'], tr[n], color=INK, lw=1.6, label='truth')
            for titr, ls, lw, a in ((False, '--', 1.3, 0.9), (True, '-', 1.8, 1.0)):
                cv = curves[(m, titr)]
                if cv is not None:
                    ax.plot(EG, cv[n], color=col, ls=ls, lw=lw, alpha=a, label='trained on titrated arms' if titr else 'trained on randomised arms')
            doses(ax, times, A[p], ymax); style(ax, ymax, xlabel=(r == len(MODELS) - 1))
            if curves[(m, False)] is None:
                ax.text(0.5, 0.5, 'EB-VAE rerun with saved curves pending\n(queue_ebvae_curves.sh)', transform=ax.transAxes, ha='center', fontsize=8, color=INK2)
            if c == 0:
                ax.set_ylabel(f'{lab}\nconcentration (mg/L)')
            if r == 0:
                ax.set_title({'amt350': 'PREDICTED: 350 mg x4', 'amt850': 'PREDICTED: 850 mg x4'}[n])
                ax.legend(fontsize=7.5, frameon=False, loc='upper left')
        ax = axes[r, 2]
        ax.plot([350, 600, 850], auc_t, color=INK, marker='o', lw=1.6, label='truth')
        for titr, ls, mk in ((False, '--', 'o'), (True, '-', 'o')):
            cv = curves[(m, titr)]
            if cv is not None:
                ax.plot([350, 600, 850], [np.trapezoid(cv[n], EG) for n in names], color=col, ls=ls, marker=mk, mfc=SURF if not titr else col,
                        lw=1.6, label='titrated' if titr else 'randomised')
        ax.set_xticks([350, 600, 850]); ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True); ax.set_ylabel('AUC 0-30 h (mg.h/L)')
        if r == 0:
            ax.set_title('DOSE RESPONSE of this patient'); ax.legend(fontsize=7.5, frameon=False, loc='upper left')
        if r == len(MODELS) - 1:
            ax.set_xlabel('dose (mg, x4)')
    fig.suptitle(f'One test patient (ke {g["P_te"][p, 1]:.2f} /h, observed on 400 mg x4; the same 6 samples go into every model): '
                 'trained on randomised vs titrated arms', x=0.01, ha='left', fontsize=10, fontweight='bold', color=INK)
    fig.text(0.01, -0.01, 'Titrated training = fast eliminators were given 800 mg in the training set; the test patient and its input are identical '
             'in both cases. Sparse N100/arm, seed 1; patient = median ke among test patients on 400 mg. Lu iScience as published '
             '(default settings, encoder fed the target regimen). Lu NMI not shown (did not learn a dose response).', fontsize=7.5, color=INK2, wrap=True)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/ex3_test2.{ext}', dpi=200, bbox_inches='tight')
    print('ex3 ->', f'{OUT}/ex3_test2.png')


if __name__ == '__main__':
    which = sys.argv[1:] or ['1', '2', '3']
    for w in which:
        {'1': fig_setup, '2': fig_test1, '3': fig_test2}[w]()
