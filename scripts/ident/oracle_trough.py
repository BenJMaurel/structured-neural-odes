#!/usr/bin/env python3
"""Trough-only oracle check: how much does the SHAPE of the prior matter when the CYP3A5
genotype is unmeasured and sampling is sparse? Upper bound on what any learned (network) prior could gain over the
Gaussian MAP-BE, before training anything.

Cohort: windowed scenario-4 benchmark (EBE_COHORT, default confound_vc00_s4_pnoise_win; 1100 test patients), TRUE
simulator population model (ebe_popmodel.spec_true: structure, covariates, omegas, Km-Vc correlation, paper noise).
CYP3A5 expressers have Vmax x 2 (theta5_CL_CYP), so with the genotype unknown log Vmax is a two-component mixture.
Priors, all with the true structure and the other covariates (formulation, hematocrit):
  known  : genotype known (= the usual true-model MAP-BE; with 3 samples it must reproduce popebe_true_pnoise_win)
  gauss  : genotype unknown, best Gaussian approximation of the mixture: log Vmax centred at p*log 2, omega^2 + p(1-p)(log 2)^2
  mixture: genotype unknown, TRUE mixture prior: MAP under each genotype, weights by Laplace evidence
           (pi_g exp(-0.5 * [sum((y-f)/sd)^2 + 2 sum log sd + eta'P eta]) det(H_g)^-1/2); 'mix_mean' = evidence-weighted
           AUC, 'mix_mode' = AUC of the more probable genotype. p = pi_expresser = share of expressers in the test set.
Designs: trough only (t = 0, pre-dose) and the 3 encoder samples (0, 1, 3 h).
Targets: V1 AUC (observed dose) and V2 AUC at the new in-range dose (same cohort). Output CSV + text summary.
    oracle_trough.py <n> <seed> <out_prefix>
"""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, 'scripts/ident'); sys.path.insert(0, 'scripts/monolix'); sys.path.insert(0, '.')
from ebe_s4_batch import simulate, FIT
from ebe_popmodel import spec_true, typical, prec, load, curve, REL, COHORT
IV = FIT.index('Vmax'); LOG2 = np.log(2.0)


def fit_group(y, tv, dose, form, PR, a, b, T, iters=40, h=1e-3):
    """MAP of eta (same Levenberg-Marquardt as ebe_popmodel.fit_group, observation times T). Returns eta,
    -2 log p(y, eta) up to a constant (incl. the residual-error log-sd terms) and log det of the GN Hessian."""
    K, P, nT = y.shape[0], len(FIT), len(T); cl = np.full(K, 20.0)
    eta = np.zeros((K, P)); lam = np.full(K, 1e-2)
    def objective(e):
        f = simulate(tv * np.exp(e), cl, dose, form, T).T; sd = a + b * f
        return (((y - f) / sd) ** 2).sum(1) + np.einsum('ki,ij,kj->k', e, PR, e)
    obj = objective(eta)
    for it in range(iters):
        pert = eta[:, None, :] + h * np.concatenate([np.zeros((1, P)), np.eye(P)])[None]
        F = simulate((tv[:, None, :] * np.exp(pert)).reshape(-1, P), np.repeat(cl, P + 1), np.repeat(dose, P + 1),
                     form, T).T.reshape(K, P + 1, nT)
        f0 = F[:, 0]; J = (F[:, 1:] - f0[:, None]) / h; W = 1.0 / (a + b * f0)
        Jw = J * W[:, None, :]; rw = (y - f0) * W
        A = np.einsum('kip,kjp->kij', Jw, Jw) + PR[None]; g = np.einsum('kip,kp->ki', Jw, rw) - eta @ PR
        step = np.linalg.solve(A + lam[:, None, None] * np.eye(P)[None], g[..., None])[..., 0]
        new = eta + step; nobj = objective(new); ok = nobj < obj
        eta[ok] = new[ok]; obj[ok] = nobj[ok]; lam = np.where(ok, lam * 0.3, lam * 10.0)
        if np.max(np.abs(step[ok]), initial=0) < 1e-5 and it > 5: break
    # evidence pieces at the mode
    F = simulate((tv[:, None, :] * np.exp(eta[:, None, :] + h * np.concatenate([np.zeros((1, P)), np.eye(P)])[None])).reshape(-1, P),
                 np.repeat(cl, P + 1), np.repeat(dose, P + 1), form, T).T.reshape(K, P + 1, nT)
    f0 = F[:, 0]; J = (F[:, 1:] - f0[:, None]) / h; sd = a + b * f0; W = 1.0 / sd
    A = np.einsum('kip,kjp->kij', J * W[:, None, :], J * W[:, None, :]) + PR[None]
    m2lp = (((y - f0) / sd) ** 2).sum(1) + 2 * np.log(sd).sum(1) + np.einsum('ki,ij,kj->k', eta, PR, eta)
    return eta, m2lp, np.linalg.slogdet(A)[1]


def auc(th, dose, form):
    hi = 12.0 if form == 'Prograf' else 24.0; ta = np.round(np.arange(0.0, hi + 1e-9, 0.02), 4)
    return np.trapezoid(simulate(th, np.full(len(dose), 20.0), dose, form, ta), ta, axis=0)


def main():
    n, seed, prefix = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    S = spec_true(); PR = prec(S)
    T0, O0 = load(COHORT)
    te = T0[T0.split == 'test'].index.values.copy(); np.random.RandomState(seed).shuffle(te); ids = te[:n]
    p = float(np.mean([T0.loc[i].CYP == 'expresser' for i in ids]))
    Sg = dict(S); Sg['om'] = S['om'].copy(); Sg['om'][IV] = np.sqrt(S['om'][IV] ** 2 + p * (1 - p) * LOG2 ** 2); PRg = prec(Sg)
    print(f'{COHORT}: {len(ids)} test patients, share expressers p = {p:.3f}; omega_Vmax {S["om"][IV]:.3f} -> Gaussian {Sg["om"][IV]:.3f}', flush=True)
    rows = []
    for form in ('Advagraf', 'Prograf'):
        g = [i for i in ids if T0.loc[i].DRUG == form]; st = 1.0 if form == 'Prograf' else 0.0
        ht = np.array([float(T0.loc[i].HT) for i in g]); cyp = np.array([int(T0.loc[i].CYP == 'expresser') for i in g])
        tv = lambda c: np.stack([typical(S, st, c[k], ht[k]) for k in range(len(g))])
        d1 = np.array([T0.loc[i].d1 for i in g]); d2 = np.array([T0.loc[i].d2 for i in g])
        full = np.array([curve(O0, i, 1)[0] for i in g]); a1 = np.array([curve(O0, i, 1)[1] for i in g]); a2 = np.array([curve(O0, i, 2)[1] for i in g])
        base = len(rows); rows += [dict(ID=i, form=form, cyp=cyp[k], d1=d1[k], d2=d2[k], auc_true_v1=a1[k], auc_true_in=a2[k]) for k, i in enumerate(g)]
        for dname, T in (('trough', [0.0]), ('three', [0.0, 1.0, 3.0])):
            y = np.stack([full[:, int(np.argmin(np.abs(REL - t)))] for t in T], 1)
            pred = {}
            for pname, TV, P_ in (('known', tv(cyp), PR), ('gauss', tv(np.full(len(g), p)), PRg)):
                eta, _, _ = fit_group(y, TV, d1, form, P_, S['a'], S['b'], T); th = TV * np.exp(eta)
                pred[pname] = (auc(th, d1, form), auc(th, d2, form))
            comp = {}
            for c in (0, 1):
                TV = tv(np.full(len(g), c)); eta, m2lp, ld = fit_group(y, TV, d1, form, PR, S['a'], S['b'], T); th = TV * np.exp(eta)
                comp[c] = (auc(th, d1, form), auc(th, d2, form), np.log(p if c else 1 - p) - 0.5 * m2lp - 0.5 * ld)
            lw = np.stack([comp[0][2], comp[1][2]], 1); w1 = np.exp(lw[:, 1] - np.logaddexp(lw[:, 0], lw[:, 1]))
            pred['mix_mean'] = tuple((1 - w1) * comp[0][j] + w1 * comp[1][j] for j in (0, 1))
            pred['mix_mode'] = tuple(np.where(w1 > 0.5, comp[1][j], comp[0][j]) for j in (0, 1))
            for k in range(len(g)):
                r = rows[base + k]; r[f'{dname}_w_expresser'] = w1[k]
                for pname, (pv1, pin) in pred.items():
                    r[f'{dname}_{pname}_v1'] = pv1[k]; r[f'{dname}_{pname}_in'] = pin[k]
            print(f'  {form} {dname}: done ({len(g)} patients)', flush=True)
    D = pd.DataFrame(rows); D.to_csv(f'{prefix}.csv', index=False)
    out = []
    for dname in ('trough', 'three'):
        out.append(f'=== design: {dname} ({"pre-dose trough only" if dname == "trough" else "0, 1, 3 h"})')
        acc = D[f'{dname}_w_expresser'].gt(0.5).eq(D.cyp.eq(1)).mean()
        out.append(f'  genotype recovered by the mixture posterior (w > 0.5): {100 * acc:.1f}% of patients')
        for tgt, lab in (('v1', 'V1 AUC (observed dose)'), ('in', 'V2 AUC (new dose, 1-8 mg)')):
            out.append(f'  -- {lab}: median |err| / bias / within 20% / RMSPE w/o worst 1% / n>100%   [by genotype: non-exp | exp median |err|]')
            for pname, plab in (('known', 'genotype known (ceiling)'), ('gauss', 'genotype unknown, Gaussian prior'),
                                ('mix_mean', 'genotype unknown, TRUE mixture (mean)'), ('mix_mode', 'genotype unknown, TRUE mixture (mode)')):
                r = D[f'{dname}_{pname}_{tgt}'] / D[f'auc_true_{tgt}'] - 1; a = r.abs(); keep = a <= a.quantile(.99)
                by = [100 * a[D.cyp == c].median() for c in (0, 1)]
                out.append(f'     {plab:40s} {100 * a.median():6.1f} {100 * r.median():+6.1f} {100 * (a <= .2).mean():6.1f} '
                           f'{100 * np.sqrt((r[keep] ** 2).mean()):7.1f} {int((a > 1).sum()):4d}   [{by[0]:5.1f} | {by[1]:5.1f}]')
    txt = '\n'.join(out); print(txt); open(f'{prefix}.txt', 'w').write(txt + '\n'); print('ORACLE_DONE', prefix)


if __name__ == '__main__':
    main()
