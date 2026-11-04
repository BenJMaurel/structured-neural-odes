"""Tacrolimus version of Baaz et al.'s regimen benchmark. Same interface as third_party/baaz_ebvae/ours/regimen/gen_regimens.py (REGIMENS, draw, simulate, noisy,
titrate, T_END), so dh_regimen.py --sim tacro runs the same training and evaluation code on it.

PK = the paper's tacrolimus simulator (gen_tacro_film.TacrolimusPK, scenario 4), Prograf only: depot -> three transit
compartments (Ktr) -> central, two-compartment disposition (Q, Vp), Michaelis-Menten elimination Vmax C / (Km + C) from the
central compartment; Vmax = 0.212 mg/h x (Ht/35)^-3.14 x 2^CYP3A5, Km = 10 ng/mL x Ht/35, eta_Km-eta_Vc correlation -0.7 with
Km's marginal variance kept (as TacrolimusPK); hematocrit uniform on 25-45 %, CYP3A5 expresser with probability 0.5.
Residual error of the published model: sd = 0.71 ng/mL + 11.3 % x C (combined linearly, as generate_patient_visit), clipped >= 0.
Integrated by classical RK4 with dt = 0.01 h, vectorised over patients (checked against TacrolimusPK's adaptive odeint, see
__main__).

Design = Baaz's, with the time axis scaled by the dosing interval (8 h -> 12 h, Prograf twice daily) and the amounts by the
arm (400 / 800 mg -> LOW / HIGH mg):
  patients start WITHOUT drug (initiation; Baaz's start with drug on board, a feature of their simulator);
  training arms LOW or HIGH mg at 0, 12, 24, 36 h; window 0-45 h (Baaz: 0-30 h);
  sparse samples 1, 3, 11.25, 23.25, 35.25, 42 h (Baaz 0, 2, 7.5, 15.5, 23.5, 28 x 1.5; the 0-h sample, zero without drug,
  replaced by 1 h); dense every 0.75 h (61 samples, as Baaz);
  unseen regimens: 0.875 / 1.5 / 2.125 x LOW on the training schedule (Baaz 350 / 600 / 850 mg) and LOW or HIGH at 0, 4.5,
  12 h (Baaz 400 / 800 mg at 0, 3, 8 h).
Covariates (CYP3A5, hematocrit) are returned with the parameters; the models receive them as static inputs.

TACRO_SCENARIO=2 in the environment (opt-in 13/10, unset = scenario 4 above, unchanged): the paper's scenario 2,
LINEAR elimination CL x C with CL = 21.2 L/h x (Ht/35)^-3.14 x 2^CYP3A5 x exp(eta_CL), omega^2 0.08 (TacrolimusPK scenario 2),
hematocrit uniform on 25-45 %, no Michaelis-Menten, no Km-Vc correlation. Column 4 of the parameter array then holds CL (L/h) and
column 5 is unused (1.0). Same design, arms and noise. Every script that imports this module picks it up (dh_regimen.py,
lu_regimen.py, run_regimen_tacro.py)."""
import os
import numpy as np

SCENARIO = int(os.environ.get('TACRO_SCENARIO', '4'))
assert SCENARIO in (2, 4), 'TACRO_SCENARIO must be 2 or 4'

LOW, HIGH = 2.0, 4.0                        # mg per Prograf dose (set from the exposure check)
ARMS = (LOW, HIGH)
DT, T_END = 0.01, 45.0
TRAIN_TIMES = [0, 12, 24, 36]
SAMP_SPARSE = np.array([1.0, 3.0, 11.25, 23.25, 35.25, 42.0])
SAMP_DENSE = np.arange(0, T_END + 1e-9, 0.75)
# 'visit' (opt-in 14/10): the INTENSIVE PK VISIT of a tacrolimus study, as the paper's development cohort -- routine
# troughs (C0, 15 min before doses 2, 3, 4) + the cohort's profile after the last dose (20, 40 min, 1, 1.5, 2, 3, 4, 6, 9 h; its 12-h
# point falls outside the 45-h window): 12 samples, between sparse (6) and dense (61); nothing observed during the first day.
SAMP_VISIT = np.array([11.75, 23.75, 35.75] + [36.0 + r for r in (1 / 3, 2 / 3, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0)])
# 'visit1' (opt-in 22/10): the same 12 samples with the intensive
# profile after the FIRST dose (20, 40 min, 1, 1.5, 2, 3, 4, 6, 9 h) and the troughs 15 min before doses 2, 3, 4.
SAMP_VISIT1 = np.array([r for r in (1 / 3, 2 / 3, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0)] + [11.75, 23.75, 35.75])
SAMP_BY_DESIGN = {'sparse': SAMP_SPARSE, 'dense': SAMP_DENSE, 'visit': SAMP_VISIT, 'visit1': SAMP_VISIT1}
EG = np.arange(0, T_END + 1e-4, 0.25)       # evaluation grid (Baaz: 0.25 h)
ADD_E, PROP_E = 0.71, 0.113                  # ng/mL, fraction (published model)
REGIMENS = {   # name: (times, amounts) -- amounts None = the patient's training arm
    'train': (TRAIN_TIMES, None),
    'amt1.75': (TRAIN_TIMES, [0.875 * LOW] * 4), 'amt3': (TRAIN_TIMES, [1.5 * LOW] * 4), 'amt4.25': (TRAIN_TIMES, [2.125 * LOW] * 4),
    'timeLOW': ([0, 4.5, 12], [LOW] * 3), 'timeHIGH': ([0, 4.5, 12], [HIGH] * 3),
}
UNSEEN = tuple(k for k in REGIMENS if k != 'train')
# columns of the parameter array
COLS = ('Ktr', 'Q', 'Vc', 'Vp', 'Vmax', 'Km', 'CYP', 'HT') if SCENARIO == 4 else ('Ktr', 'Q', 'Vc', 'Vp', 'CL', 'unused', 'CYP', 'HT')
PROBE = {'log Vc': 2, 'log Vmax': 4, 'log Km': 5} if SCENARIO == 4 else {'log Vc': 2, 'log CL': 4}   # latent probes (Lu identity test)
_TV = dict(Ktr=3.34 * 1.53, Q=79.0, Vc=486.0 * 0.29, Vp=271.0, Vmax=21.2 * 0.01, Km=0.01)   # Prograf typical values (mg, L, h)
_OM = dict(Ktr=np.sqrt(0.06), Q=np.sqrt(0.29), Vc=np.sqrt(0.10), Vp=np.sqrt(0.36), Vmax=np.sqrt(0.08),
           Km=float(np.sqrt(max(0.10 - (0.1683 * 1.0) ** 2, 1e-6))))
RHO_KM_VC = -0.7


def draw(n, rng):
    """[n, 8] array, columns COLS (Km in mg/L, HT in %, CYP 0/1)"""
    cyp = (rng.random(n) < 0.5).astype(float)
    ht = rng.uniform(25.0, 45.0, n)
    if SCENARIO == 2:
        eta = {k: rng.standard_normal(n) * w for k, w in (('Ktr', _OM['Ktr']), ('Q', _OM['Q']), ('Vc', _OM['Vc']), ('Vp', _OM['Vp']),
                                                           ('CL', np.sqrt(0.08)))}
        cl = 21.2 * (ht / 35.0) ** -3.14 * 2.0 ** cyp * np.exp(eta['CL'])
        return np.stack([_TV['Ktr'] * np.exp(eta['Ktr']), _TV['Q'] * np.exp(eta['Q']), _TV['Vc'] * np.exp(eta['Vc']),
                         _TV['Vp'] * np.exp(eta['Vp']), cl, np.ones(n), cyp, ht], 1)
    eta = {k: rng.standard_normal(n) * _OM[k] for k in ('Ktr', 'Q', 'Vc', 'Vp', 'Vmax', 'Km')}
    eta['Km'] = RHO_KM_VC * (_OM['Km'] / _OM['Vc']) * eta['Vc'] + np.sqrt(1 - RHO_KM_VC ** 2) * eta['Km']
    vmax = _TV['Vmax'] * (ht / 35.0) ** -3.14 * 2.0 ** cyp * np.exp(eta['Vmax'])
    km = _TV['Km'] * (ht / 35.0) * np.exp(eta['Km'])
    return np.stack([_TV['Ktr'] * np.exp(eta['Ktr']), _TV['Q'] * np.exp(eta['Q']), _TV['Vc'] * np.exp(eta['Vc']),
                     _TV['Vp'] * np.exp(eta['Vp']), vmax, km, cyp, ht], 1)


def _rhs(A, P):
    ktr, q, vc, vp, vmax, km = (P[:, i] for i in range(6))
    d, g1, g2, g3, c, p = A
    C = c / vc
    elim = vmax * C if SCENARIO == 2 else vmax * C / (km + C)      # scenario 2: column 4 = CL (L/h), linear elimination
    return np.stack([-ktr * d, ktr * (d - g1), ktr * (g1 - g2), ktr * (g2 - g3),
                     ktr * g3 - elim - q / vc * c + q / vp * p, q / vc * c - q / vp * p])


def simulate(params, times, amounts):
    """params [n, 8]; times list (h); amounts [n, n_doses] mg -> (t_eval, concentrations ng/mL [n, T]) on the DT grid.
    Drug-free at t = 0; a dose at t goes into the depot before the output at t (the central concentration is continuous)."""
    P = np.asarray(params, float); n = len(P)
    t_eval = np.arange(0.0, T_END + DT / 2, DT)
    A = np.zeros((6, n)); out = np.zeros((n, len(t_eval))); k = 0
    for i, t in enumerate(t_eval):
        while k < len(times) and t >= times[k] - 1e-9:
            A[0] = A[0] + amounts[:, k]; k += 1
        out[:, i] = A[4] / P[:, 2] * 1000.0
        k1 = _rhs(A, P); k2 = _rhs(A + DT / 2 * k1, P); k3 = _rhs(A + DT / 2 * k2, P); k4 = _rhs(A + DT * k3, P)
        A = np.maximum(A + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
    return t_eval, out


def noisy(c, rng):
    return np.maximum(0.0, c + (ADD_E + PROP_E * c) * rng.standard_normal(c.shape))


def titrate(P, rho, seed):
    """titrated arms: patients with a high elimination capacity (log Vmax; log CL in scenario 2) get the HIGH dose. score = rho z(log Vmax) +
    sqrt(1 - rho^2) N(0, 1) from its own stream (seed + 9001), top half -> HIGH (balanced). Returns (arm, corr(log Vmax, HIGH))."""
    lv = np.log(P[:, 4]); z = (lv - lv.mean()) / lv.std()
    score = rho * z + np.sqrt(1 - rho ** 2) * np.random.default_rng(seed + 9001).standard_normal(len(P))
    arm = np.where(score > np.median(score), HIGH, LOW)
    return arm, float(np.corrcoef(lv, arm == HIGH)[0, 1])


if __name__ == '__main__':   # check against TacrolimusPK (adaptive odeint, scenario 4 forward) and print the exposure
    import os, sys, torch
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from gen_tacro_film import TacrolimusPK
    rng = np.random.default_rng(0); P = draw(2000, rng)
    worst = 0.0
    for i in range(8):
        pk = TacrolimusPK(formulation='Prograf', scenario=SCENARIO)
        pk.individual_params = {k: torch.tensor(float(P[i, j])) for j, k in enumerate(COLS[:6]) if k != 'unused'}
        if SCENARIO == 4:
            pk.individual_params['CL'] = pk.individual_params['Vmax'] / pk.individual_params['Km']   # inert under Michaelis-Menten
        pk.dose_mg = HIGH
        # the dose times must be in tq: TacrolimusPK.simulate applies a dose at the last requested time <= the dose time
        tq = torch.tensor([1.0, 3.0, 11.25, 12.0, 23.25, 24.0, 30.0, 35.25, 36.0, 42.0, 45.0])
        ref = pk.simulate([0.0, 12.0, 24.0, 36.0], tq, resample=False).reshape(-1)[1:].numpy() * 1000.0
        t_eval, c = simulate(P[i:i + 1], TRAIN_TIMES, np.full((1, 4), HIGH))
        mine = np.interp(tq.numpy(), t_eval, c[0])
        worst = max(worst, float(np.max(np.abs(mine / ref - 1))))
    print(f'scenario {SCENARIO}: max relative difference vs TacrolimusPK (8 patients, 9 times, odeint tol 1e-6): {worst:.2e}')
    for nm, S in (('sparse', SAMP_SPARSE), ('dense', SAMP_DENSE)):   # the fixed-noise rule: 2 x RMS residual sd on the training samples
        arm = np.where(np.arange(200) % 2 == 0, LOW, HIGH); t_, c_ = simulate(P[:200], TRAIN_TIMES, np.repeat(arm[:, None], 4, 1))
        C_ = np.stack([np.interp(S, t_, ci) for ci in c_]); print(f'{nm}: RMS residual sd {np.sqrt(np.mean((ADD_E + PROP_E * C_) ** 2)):.2f} ng/mL')
    for name, (times, amts) in REGIMENS.items():
        A = np.tile(np.array(amts if amts else [LOW] * 4, float), (len(P), 1))
        t_eval, c = simulate(P, times, A)
        last = (t_eval >= 36) & (t_eval <= 45)
        cav = np.trapezoid(c[:, last], t_eval[last], axis=1) / 9.0
        print(f'{name:9s} AUC(0-45) median {np.median(np.trapezoid(c, t_eval, axis=1)):7.1f}  Cavg(36-45 h) median {np.median(cav):5.1f} '
              f'[{np.percentile(cav, 5):.1f}, {np.percentile(cav, 95):.1f}] ng/mL')
    for d in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        t_eval, c = simulate(P, TRAIN_TIMES, np.full((len(P), 4), d))
        a = np.trapezoid(c, t_eval, axis=1)
        print(f'{d:3.0f} mg q12h: AUC(0-45) median {np.median(a):7.1f}  per mg {np.median(a) / d:6.1f}  trough 35.25 h median '
              f'{np.median(c[:, int(round(35.25 / DT))]):5.1f} ng/mL')
