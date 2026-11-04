#!/usr/bin/env python3
"""Null test of the tracker under the TRUE model -- the reference for track_null.py.

    EBE_COHORT=confound_vc00_s4_pnoise_win track_null_truemodel.py <out.json> [--dref 4] [--n 0]

Per-visit MAP-BE ("IND") with the true scenario-4 structure, the true population model and the true residual error
(scripts/monolix/ebe_popmodel.py 'true' spec and its Levenberg-Marquardt), fitted SEPARATELY to the 0/1/3 h samples of
V1 and of V2 of the same test patients as track_null.py. Readout: log AUC at d_ref mg (own formulation) with each
visit's EBE. Truth: no movement (same parameters at both visits). Laplace sd from the Gauss-Newton Hessian (prior
included) and a finite-difference gradient of the readout. Whatever spread or dose-ratio slope appears here comes
from sampling noise and prior shrinkage alone -- the floor a correct model would show with the same tracker.
"""
import os, sys, json, argparse
import numpy as np
os.environ.setdefault('EBE_COHORT', 'confound_vc00_s4_pnoise_win')
sys.path.insert(0, 'scripts/monolix'); sys.path.insert(0, 'scripts/ident'); sys.path.insert(0, 'scripts/tracking')
sys.path.insert(0, '.')
import ebe_popmodel as EP
from ebe_s4_batch import simulate, ENC_T, FIT
from mle_km_s4 import replay
from track_null import summarize, log


def fit_with_hessian(y, tv, cl, dose, form, PR, a, b, iters=40, h=1e-3):
    """ebe_popmodel.fit_group (same objective, steps and stopping rule) + the Gauss-Newton Hessian at the solution."""
    eta = EP.fit_group(y, tv, cl, dose, form, PR, a, b, iters=iters, h=h)
    K = y.shape[0]
    pert = eta[:, None, :] + h * np.concatenate([np.zeros((1, 6)), np.eye(6)])[None]
    F = simulate((tv[:, None, :] * np.exp(pert)).reshape(-1, 6), np.repeat(cl, 7), np.repeat(dose, 7), form,
                 ENC_T).T.reshape(K, 7, 3)
    f0 = F[:, 0]
    Jw = (F[:, 1:] - f0[:, None]) / h / (a + b * f0)[:, None, :]
    A = np.einsum('kip,kjp->kij', Jw, Jw) + PR[None]
    return eta, A, f0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--dref', type=float, default=4.0)
    ap.add_argument('--n', type=int, default=0)
    cli = ap.parse_args()
    S = EP.spec_true()
    PR = EP.prec(S)
    T0, O0 = EP.load(EP.COHORT)
    if all(c in T0.columns for c in FIT):
        # cohorts that record their parameters (e.g. --v2-shift): use them, as ebe_popmodel.main does
        TP = T0.rename(columns={'DRUG': 'form'}).copy()
        TP['cyp'] = (TP['CYP'] == 'expresser').astype(int)
    else:
        TP = replay(max(1800, int(T0.index.max())), 4)
    te = list(T0[T0.split == 'test'].index.values)
    if cli.n:
        te = te[:cli.n]
    log(f'cohort {EP.COHORT}: {len(te)} test patients; residual sd {S["a"]:g} + {S["b"]:g}*C; d_ref {cli.dref:g} mg')
    out = {k: [] for k in ('ids', 'dose_v1_mg', 'dose_v2_mg', 'prograf', 'true_auc_v1', 'true_auc_v2', 'g1', 'g2', 'sd',
                           'g_true', 'auc_fit_v1', 'auc_fit_v2', 'auc_cf_v2_from_v1')}
    for form in ('Advagraf', 'Prograf'):
        g = [i for i in te if TP.loc[i].form == form]
        if not g:
            continue
        st = 1.0 if form == 'Prograf' else 0.0
        TV = np.stack([EP.typical(S, st, int(TP.loc[i].cyp), float(TP.loc[i].HT)) for i in g])
        CL = np.full(len(g), 20.0)                      # inert under MM: never enters the ODE (pitfall )
        d1 = np.array([T0.loc[i].d1 for i in g])
        d2 = np.array([T0.loc[i].d2 for i in g])
        y = {}
        for v in (1, 2):
            y[v] = np.array([[EP.curve(O0, i, v)[0][int(np.argmin(np.abs(EP.REL - t)))] for t in ENC_T] for i in g])
        hi = 12.0 if form == 'Prograf' else 24.0
        ta = np.round(np.arange(0.0, hi + 1e-9, 0.02), 4)
        dref = np.full(len(g), cli.dref)

        def logauc(theta, dose):
            return np.log(np.trapezoid(simulate(theta, np.full(len(theta), 20.0), dose, form, ta), ta, axis=0))

        def readout(eta):
            """log AUC at d_ref and its finite-difference gradient in eta."""
            K = eta.shape[0]
            pert = eta[:, None, :] + 1e-3 * np.concatenate([np.zeros((1, 6)), np.eye(6)])[None]
            la = logauc((TV[:, None, :] * np.exp(pert)).reshape(-1, 6), np.repeat(dref, 7)).reshape(K, 7)
            return la[:, 0], (la[:, 1:] - la[:, :1]) / 1e-3

        log(f'{form}: {len(g)} patients, fitting V1')
        e1, A1, _ = fit_with_hessian(y[1], TV, CL, d1, form, PR, S['a'], S['b'])
        log(f'{form}: fitting V2')
        e2, A2, _ = fit_with_hessian(y[2], TV, CL, d2, form, PR, S['a'], S['b'])
        g1, gr1 = readout(e1)
        g2, gr2 = readout(e2)
        sd = np.sqrt(np.einsum('ki,kij,kj->k', gr1, np.linalg.inv(A1), gr1) + np.einsum('ki,kij,kj->k', gr2, np.linalg.inv(A2), gr2))
        th_true = np.stack([[float(TP.loc[i][nm]) for nm in FIT] for i in g])
        g_true = logauc(th_true, dref)
        if 'v2_shift_param' in T0.columns:
            # --v2-shift cohort: visit-2 truth = visit-1 parameters with the shifted one replaced
            prm = str(T0['v2_shift_param'].dropna().iloc[0])
            th2 = th_true.copy()
            th2[:, FIT.index(prm)] = [float(T0.loc[i][f'{prm}_v2']) for i in g]
            g_true_v2 = logauc(th2, dref)
        else:
            g_true_v2 = g_true
        th1, th2 = TV * np.exp(e1), TV * np.exp(e2)
        auc = lambda th, dd: np.exp(logauc(th, dd))
        out['ids'] += [str(int(i)) for i in g]
        out['dose_v1_mg'] += d1.tolist(); out['dose_v2_mg'] += d2.tolist(); out['prograf'] += [int(st)] * len(g)
        out['true_auc_v1'] += [EP.curve(O0, i, 1)[1] for i in g]; out['true_auc_v2'] += [EP.curve(O0, i, 2)[1] for i in g]
        out['g1'] += g1.tolist(); out['g2'] += g2.tolist(); out['sd'] += sd.tolist(); out['g_true'] += g_true.tolist()
        out.setdefault('g_true_v2', []).extend(g_true_v2.tolist())
        out['auc_fit_v1'] += auc(th1, d1).tolist(); out['auc_fit_v2'] += auc(th2, d2).tolist()
        out['auc_cf_v2_from_v1'] += auc(th1, d2).tolist()
        log(f'{form}: done')
    g1, g2, sd = (np.array(out[k]) for k in ('g1', 'g2', 'sd'))
    xlr = np.log(np.array(out['dose_v2_mg']) / np.array(out['dose_v1_mg']))
    s = summarize(g2 - g1, xlr, sd)
    for k, tk in (('auc_fit_v1', 'true_auc_v1'), ('auc_fit_v2', 'true_auc_v2'), ('auc_cf_v2_from_v1', 'true_auc_v2')):
        s[f'{k}_median_abs_err_pct'] = 100 * float(np.median(np.abs(np.array(out[k]) / np.array(out[tk]) - 1)))
    s['readout_err_v1_median_abs_pct'] = 100 * float(np.median(np.abs(g1 - np.array(out['g_true']))))
    res = {'model': 'true structure + true population model (MAP-BE)', 'cohort': EP.COHORT, 'dref_mg': cli.dref,
           'n': len(out['ids']), 'summary': s, **out, 'delta': (g2 - g1).tolist()}
    log('true model  |Delta| median {median_abs_pct:5.1f}% p90 {p90_abs_pct:5.1f}% mean {mean_pct:+5.1f}% slope '
        '{slope_on_log_dose_ratio:+.3f} (se {slope_se:.3f}) flags z {flag_z196:.3f} >20% {flag_p_gt20_095:.3f} | fit V1 '
        '{auc_fit_v1_median_abs_err_pct:.1f}% V2 {auc_fit_v2_median_abs_err_pct:.1f}% cf V2 '
        '{auc_cf_v2_from_v1_median_abs_err_pct:.1f}%'.format(**s))
    os.makedirs(os.path.dirname(cli.out) or '.', exist_ok=True)
    json.dump(res, open(cli.out, 'w'))
    log(f'wrote {cli.out}')


if __name__ == '__main__':
    main()
