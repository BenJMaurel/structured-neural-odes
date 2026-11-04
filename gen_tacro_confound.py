###########################
# Exp 2: confounded dose assignment.
#
# Hypothesis under test: because dose-cond feeds d into the vector field at every
# step, it can use d as a SHORTCUT for elimination rate whenever the two are
# correlated in the training data -- learning "high dose => fast decay" instead of
# reading clearance out of the latent state. OT-FiLM's field is autonomous, so d
# cannot modulate downstream rates mid-course and the shortcut is unavailable.
#
# The confound is the clinically realistic one: clinicians titrate high-clearance
# patients UP to reach target exposure, so in observational data dose correlates
# with clearance.
#
#   dose index = Phi( rho * z_CL + sqrt(1-rho^2) * eps )  mapped onto the 7-level grid
#   with z_CL the patient's standardised log baseline clearance.
#
# rho = 0 reproduces the published independent assignment exactly.
#
# TWO RNG STREAMS: patient physiology is drawn from `seed`, dose assignment from
# `dose_seed`. Two cohorts generated with the same `seed` but different rho contain
# THE SAME PATIENTS and differ only in how doses were assigned -- which is what
# makes the confounded-vs-independent comparison an isolated one.
#
# The test split's visit-2 dose is drawn INDEPENDENTLY of clearance (a randomised
# intervention), so a model that leaned on the shortcut is penalised there.
###########################

import os, argparse, random
import numpy as np, pandas as pd, torch
from scipy.stats import norm
from sklearn.model_selection import train_test_split

from gen_tacro_film import TacrolimusPK, generate_patient_visit

GRID = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# Which PK parameter the dose is confounded WITH. This choice decides whether the
# confound can bite at all:
#
#   CL   -- sets the LEVEL of the curve. Measured shape range of the dose-response
#           = 0.000: changing CL fourfold does not alter how exposure scales with
#           dose at all, and the level is directly visible in the observed window.
#           Confounding on it is doubly harmless (this was the first design).
#   Km   -- sets the CURVATURE of the dose-response (shape range 0.203, the largest
#           of any parameter). Not identifiable from a single-dose window -- you need
#           >= 2 doses to see it -- and it governs exactly the quantity the
#           counterfactual scores. Clinically: CYP3A5 genotype-guided starting doses
#           are standard practice and genotype also governs metabolic saturation.
#   Vmax -- intermediate (0.131).
#   Vp   -- shapes the 4-24 h tail, which the encoder never observes (its window
#           ends at 3 h). Strong mechanism, weak clinical story.
# Vc is the confounder that makes the experiment a genuine shortcut test: unlike Km
# it IS inferable from a single-dose curve (R^2 = 0.86 from the latent on a control
# cohort), so a legitimate route exists and a model that leans on the dose instead
# has taken a shortcut rather than used its only option.  It also acts on the
# Michaelis-Menten dynamics -- C = A/Vc sets how saturated elimination is -- so it
# changes the dose-exposure CURVATURE (15.9%) and is therefore needed for the
# counterfactual rather than derivable by proportional scaling.
CONFOUND_PARAMS = ('CL', 'Km', 'Vmax', 'Vp', 'Vc')


# ---- realistic dose-switch design (opt-in, --switch-delay; 2026-10-03, Baaz comparison Part 2) ----------------------
# Default design (unchanged): each visit = the 7th dose given to an EMPTY body (nbr_ss = 6 previous doses), visits
# independent. With Michaelis-Menten elimination many patients are far from steady state after 7 doses, and some have
# none (dose rate > Vmax). Switch design: visit 1 = the EXACT periodic steady state on d1 (fixed point of the one-interval
# dose map, Newton; patients without one are rejected), observed over [0, 24] h after a dose (no extra dose inside the
# observed interval, as in the default generator); the dose switches to d2 at t = 24 h and is given every tau; visit 2 =
# the interval starting `delay` days later (t = 24 + 24*delay), observed the same way. ONE continuous trajectory, so a
# model that simulates time (e.g. the EB-VAE) sees exactly the same process. AUC = true concentration over [0, tau].
_SW_REL = [0., 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.]


def _sw_integrate(pk, state, t0, t1, out_t):
    """integrate the PK ODE from `state` (pre-dose amounts, float64 tuple) over [t0, t1]; concentrations (ng/mL) at the
    requested times in (t0, t1] and the end state."""
    want = sorted(set(float(t) for t in out_t if t0 < t <= t1 + 1e-12))
    tt = torch.tensor(sorted(set([float(t0)] + want + [float(t1)])), dtype=torch.float64)
    sol = pk.odeint(pk, state, tt, atol=1e-9, rtol=1e-9)
    vc = pk.individual_params['Vc']
    conc = {float(t): float(sol[4][k].reshape(-1)[0] / vc.reshape(-1)[0]) * 1000.0 for k, t in enumerate(tt.tolist())}
    return {t: conc[t] for t in want}, tuple(x[-1].reshape(1) for x in sol)


def _sw_dose(state, amt):
    return (state[0] + amt,) + tuple(state[1:])


def periodic_steady_state(pk, dose, tau, warm_days=14, max_newton=30):
    """pre-dose state s with Phi(s) = s (Phi = dose into the depot, then one interval tau). Returns (s, converged)."""
    z = tuple(torch.zeros(1, dtype=torch.float64) for _ in range(6))
    for _ in range(int(warm_days * 24 / tau)):
        _, z = _sw_integrate(pk, _sw_dose(z, dose), 0.0, tau, [])
    phi = lambda v: np.array([float(x) for x in _sw_integrate(pk, _sw_dose(tuple(torch.tensor([a], dtype=torch.float64) for a in v), dose), 0.0, tau, [])[1]])
    s = np.array([float(x) for x in z])
    for _ in range(max_newton):
        f = phi(s) - s
        if np.max(np.abs(f)) <= 1e-10 + 1e-8 * np.max(np.abs(s)): return s, True
        J = np.zeros((6, 6))
        for j in range(6):
            h = 1e-7 * max(abs(s[j]), 1e-6); e = s.copy(); e[j] += h; J[:, j] = (phi(e) - s - (e - s) - f) / h
        step = np.linalg.solve(J, -f); lam = 1.0
        while lam > 1e-3 and np.any(s + lam * step < -1e-12): lam *= 0.5
        s = np.maximum(s + lam * step, 0.0)
    f = phi(s) - s
    return s, bool(np.max(np.abs(f)) <= 1e-8 + 1e-6 * np.max(np.abs(s)))


def simulate_switch(pk, d1, d2, formulation, delay_days, ss=None):
    """true curves of the switch design: dict(v1, v2 = concentrations at _SW_REL, auc1, auc2, t2, ss) or None (no
    steady state). `ss` = a precomputed (state, converged) for (pk, d1) to skip the Newton search."""
    tau = 12.0 if formulation == 'Prograf' else 24.0
    s, ok = ss if ss is not None else periodic_steady_state(pk, d1, tau)
    if not ok: return None
    s_keep = np.array(s, dtype=float)
    s = tuple(torch.tensor([a], dtype=torch.float64) for a in s)
    vc = float(pk.individual_params['Vc'].reshape(-1)[0])
    fine = [round(0.01 * k, 2) for k in range(1, int(tau * 100) + 1)]
    c1, st = _sw_integrate(pk, _sw_dose(s, d1), 0.0, 24.0, _SW_REL[1:] + fine)
    v1 = [float(s[4]) / vc * 1000.0] + [c1[t] for t in _SW_REL[1:]]
    auc1 = np.trapezoid([v1[0]] + [c1[t] for t in fine], [0.0] + fine)
    t2 = 24.0 + 24.0 * delay_days; t = 24.0
    while t < t2 - 1e-9:
        _, st = _sw_integrate(pk, _sw_dose(st, d2), t, t + tau, []); t += tau
    pre2 = float(st[4]) / vc * 1000.0
    c2, _ = _sw_integrate(pk, _sw_dose(st, d2), t2, t2 + 24.0, [t2 + r for r in _SW_REL[1:]] + [t2 + f for f in fine])
    v2 = [pre2] + [c2[t2 + r] for r in _SW_REL[1:]]
    auc2 = np.trapezoid([pre2] + [c2[t2 + f] for f in fine], [0.0] + fine)
    return dict(v1=v1, v2=v2, auc1=float(auc1), auc2=float(auc2), t2=t2, ss=(s_keep, True))


def _switch_job(args):
    pk, d1, d2, form, delay, ss = args
    torch.set_num_threads(1)
    r = simulate_switch(pk, d1, d2, form, delay, ss=ss)
    if r is None:
        return None, (None, False)
    return r, r['ss']


def run_switch_jobs(jobs, cache_path=None, workers=None):
    """jobs: dict key -> (pk, d1, d2, form, delay, cache_key). Parallel over processes (env SWITCH_WORKERS, default 4);
    visit-1 steady states cached on disk by cache_key, so the out-of-range cohorts (same patients, same d1) reuse them."""
    import pickle, concurrent.futures as cf
    cache = pickle.load(open(cache_path, 'rb')) if cache_path and os.path.exists(cache_path) else {}
    workers = workers or int(os.environ.get('SWITCH_WORKERS', '4'))
    keys = list(jobs); args = [(jobs[k][0], jobs[k][1], jobs[k][2], jobs[k][3], jobs[k][4], cache.get(jobs[k][5])) for k in keys]
    hit = sum(a[5] is not None for a in args)
    print(f"switch design: {len(keys)} patients to simulate, {hit} visit-1 steady states from cache, {workers} workers", flush=True)
    out = {}
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        for n, (k, (res, ss)) in enumerate(zip(keys, ex.map(_switch_job, args, chunksize=4)), 1):
            out[k] = res; cache[jobs[k][5]] = ss
            if n % 200 == 0: print(f"  ...{n}/{len(keys)}", flush=True)
    if cache_path:
        pickle.dump(cache, open(cache_path, 'wb'))
    return out


def switch_visit_rows(pk, visit_id, patient_id, nbr_ss, conc_true, auc, dose, formulation, hematocrit, cyp_status):
    """rows in the exact format of gen_tacro_film.generate_patient_visit, for given true concentrations at _SW_REL."""
    import gen_tacro_film as _g
    true = torch.tensor(conc_true, dtype=torch.float32).reshape(-1, 1)
    noise = torch.randn_like(true)
    conc = torch.clamp(true + (_g.RESIDUAL_ERROR_ADD_SD + _g.RESIDUAL_ERROR_PROP_SD * true) * noise, min=0.0)
    ip = pk.individual_params; ii = 12. if formulation == 'Prograf' else 24.
    k_elim = float(ip['CL']) / float(ip['Vc']); k_12 = float(ip['Q']) / float(ip['Vc']); k_21 = float(ip['Q']) / float(ip['Vp'])
    ST = 1 if ii == 12. else 0; CYP = 1 if cyp_status == 'expresser' else 0
    base = dict(PERI=1, CYP=CYP, DRUG=formulation, AUC=auc, ST=ST, HT=hematocrit, K_ELIM=k_elim, K_12=k_12, K_21=k_21)
    rows = [dict(ID=patient_id, VISIT=visit_id, TIME=0.0, DV='.', AMT=dose, II=ii, nbr_ss=nbr_ss, mdv=1, ss=1, **base)]
    for i in range(nbr_ss):
        rows.append(dict(ID=patient_id, VISIT=visit_id, TIME=-ii * (i + 1), DV='.', AMT=dose, II=ii, mdv=1, ss=1, **base))
    for t, c, ct in zip(_SW_REL, conc.reshape(-1).tolist(), true.reshape(-1).tolist()):
        rows.append(dict(ID=patient_id, VISIT=visit_id, TIME=float(t), DV=(c if c != 0 else ct), AMT='.', II='.',
                         nbr_ss=nbr_ss, mdv=0, ss='.', **base))
    return rows


def draw_v2_shift(rng, dist, scale, frac):
    """--v2-shift (2026-10-06): log-scale change of one PK parameter at visit 2. Exactly two draws per
    patient from the dedicated stream, whatever happens to the patient afterwards, so a patient's change depends only on
    its position in the cohort. With probability 1 - frac no change."""
    u, v = rng.uniform(), (rng.uniform(-1.0, 1.0) if dist == 'uniform' else rng.normal())
    return 0.0 if u >= frac else float(scale * v)


def visit2_shifted(pk, param, delta, pid, nbr_ss, obs, form, ht, cyp):
    """Visit 2 with pk.individual_params[param] scaled by exp(delta), plus the AUC the patient would have had at the same
    dose WITHOUT the change. That no-change simulation runs between a save and a restore of every RNG state, so the
    observation-noise stream is consumed exactly as in a cohort without --v2-shift (identical visit-1 data, identical
    later patients). Returns (visit-2 rows, AUC without the change, the parameter's visit-2 value)."""
    st = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    auc_noshift = float(generate_patient_visit(pk, 2, pid, nbr_ss, obs, form, ht, cyp)[0]['AUC'])
    random.setstate(st[0]); np.random.set_state(st[1]); torch.set_rng_state(st[2])
    orig = pk.individual_params[param]
    pk.individual_params[param] = orig * float(np.exp(delta))
    try:
        rows = generate_patient_visit(pk, 2, pid, nbr_ss, obs, form, ht, cyp)
    finally:
        pk.individual_params[param] = orig
    return rows, auc_noshift, float(orig) * float(np.exp(delta))


def assign_dose(z_cl, rho, rng, grid=GRID):
    """Map a standardised log-clearance to a dose level with correlation ~rho."""
    u = rho * z_cl + np.sqrt(max(0.0, 1.0 - rho ** 2)) * rng.normal()
    k = int(np.clip(np.floor(norm.cdf(u) * len(grid)), 0, len(grid) - 1))
    return grid[k]


def build(num_patients, rho, scenario, seed, dose_seed, test_fraction, randomize_test, ood_v2_grid=None,
          test_v2_rho=None,
          nbr_ss=6, confound_param='CL', grid=None, zstats=None, id_offset=0,
          eta_dist='log_normal', t_df=4, ktr_slow_frac=0.0, ktr_slow_factor=1.0, v1_cavg_window=None,
          ood_keep_v1=False, sat_scale=1.0, sat_logunif=None, switch_delay=None, switch_cache=None, v2_shift=None):
    obs = torch.tensor([0, 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.]) + 24 * nbr_ss
    # v2_shift = (param, 'uniform'|'normal', scale, frac) or None (default: visits share every parameter, as before)
    assert v2_shift is None or switch_delay is None, "--v2-shift is not implemented for the --switch-delay design"
    # ---- pass 1: physiology only, so z_CL can be standardised over the cohort ----
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    slow_rng = np.random.RandomState(seed + 4242)
    # saturation reserve: Km AND Vmax scaled by the same per-patient factor k, so Vmax/Km (the low-concentration
    # clearance) is unchanged and only the degree of Michaelis-Menten saturation moves. Fixed k (sat_scale) or
    # log-uniform per patient (sat_logunif) drawn from its OWN stream, so every other stream is consumed as before.
    sat_on = (sat_scale != 1.0) or (sat_logunif is not None)
    assert not sat_on or scenario >= 3, "--sat-scale / --sat-logunif need Michaelis-Menten (scenario >= 3)"
    sat_rng = np.random.RandomState(seed + 7331)
    record_params = (eta_dist != 'log_normal') or ktr_slow_frac > 0 or sat_on or (v2_shift is not None)
    pats = []
    for pid in range(1, num_patients + 1):
        form = random.choice(['Prograf', 'Advagraf'])
        cyp = random.choice(['expresser', 'non_expresser'])
        ht = random.uniform(25.0, 45.0) if scenario in (2, 4) else 35.0
        pk = TacrolimusPK(formulation=form, hematocrit=ht, distribution_type=eta_dist,
                          cyp_status=cyp, scenario=scenario)
        pk._sample_individual_parameters(t_df=t_df)
        slow = 0
        if ktr_slow_frac > 0:
            # unrecorded slow/delayed-absorber subgroup: Ktr scaled by ktr_slow_factor. The indicator comes
            # from its OWN stream, so the random/torch streams (covariates, etas) are consumed as before.
            slow = int(slow_rng.rand() < ktr_slow_frac)
            if slow:
                pk.individual_params['Ktr'] = pk.individual_params['Ktr'] * ktr_slow_factor
        sat_k = 1.0
        if sat_logunif is not None:
            sat_k = float(np.exp(sat_rng.uniform(np.log(sat_logunif[0]), np.log(sat_logunif[1]))))
        elif sat_scale != 1.0:
            sat_k = float(sat_scale)
        if sat_on:
            pk.individual_params['Km'] = pk.individual_params['Km'] * sat_k
            pk.individual_params['Vmax'] = pk.individual_params['Vmax'] * sat_k
        pats.append(dict(pid=pid, pk=pk, form=form, cyp=cyp, ht=ht, slow=slow, sat_k=sat_k,
                         cl=float(pk.individual_params['CL']),
                         par=float(pk.individual_params[confound_param])))
    grid = grid or GRID
    # standardise the log of whichever parameter the dose is confounded with
    lp = np.log(np.array([p['par'] for p in pats]))
    if zstats is not None:
        # Pin the standardisation to a reference cohort.  z_CL feeds assign_dose, so
        # re-deriving mean/sd on a different cohort would shift every patient's dose
        # and change the realised dose-physiology correlation -- the quantity the
        # whole experiment turns on.  Pinning keeps a freshly generated held-out set
        # on the same dose scale as the cohort the models were trained on.
        zm, zs = zstats
    else:
        zm, zs = lp.mean(), lp.std()
    z_cl = (lp - zm) / zs

    # ---- pass 2: dose assignment on its own stream ----
    n_train = int(round(num_patients * (1 - test_fraction)))
    sw_results, sw_jobs = {}, {}
    for phase in (('collect', 'emit') if switch_delay is not None else ('emit',)):
      if phase == 'emit' and switch_delay is not None:
        sw_results = run_switch_jobs(sw_jobs, cache_path=switch_cache)
      rng = np.random.RandomState(dose_seed)
      rng_v2 = np.random.RandomState(dose_seed + 9173)   # only used when test_v2_rho is set
      n_rejected = [0, 0]   # [test, train] patients outside the visit-1 exposure window
      n_no_ss = [0, 0]      # [test, train] patients without a visit-1 steady state (switch design only)
      rng_ood = np.random.RandomState(dose_seed + 5151)   # only used with ood_keep_v1
      rng_shift = np.random.RandomState(seed + 6007)      # only used with v2_shift
      rows, truth = [], []
      for i, p in enumerate(pats):
          is_test = (p['pid'] > n_train)
          delta = draw_v2_shift(rng_shift, *v2_shift[1:]) if v2_shift is not None else 0.0
          sh = {}   # truth columns added by --v2-shift
          d1 = assign_dose(z_cl[i], rho, rng, grid)
          if is_test and ood_v2_grid:
              # OUT-OF-SUPPORT intervention: visit-2 dose drawn from a grid the model
              # never saw in training. visit 1 stays in-distribution so the encoder is
              # given a familiar visit -- this isolates dose extrapolation from an
              # inability to encode the patient at all.
              if ood_keep_v1:
                  # consume the dose stream exactly as the in-range cohort does (randomised d2, redrawn while equal
                  # to d1), so every later patient's d1 is identical across the in-range and out-of-range cohorts;
                  # the out-of-range d2 comes from its own stream.
                  _d = grid[rng.randint(len(grid))]
                  while _d == d1:
                      _d = grid[rng.randint(len(grid))]
                  d2 = ood_v2_grid[rng_ood.randint(len(ood_v2_grid))]
              else:
                  d2 = ood_v2_grid[rng.randint(len(ood_v2_grid))]
          elif is_test and randomize_test:
              # randomised intervention: visit-2 dose independent of clearance
              d2 = grid[rng.randint(len(grid))]
              while d2 == d1:
                  d2 = grid[rng.randint(len(grid))]
              if test_v2_rho is not None:
                  # Re-draw the counterfactual dose with a prescribed correlation to the
                  # confounded parameter, using a DEDICATED stream. The randomised draws
                  # above are still made and discarded, so `rng` is consumed identically
                  # to the default path: the cohort therefore contains the same patients,
                  # with the same visit-1 doses, and differs only in the visit-2 rule.
                  d2 = assign_dose(z_cl[i], test_v2_rho, rng_v2, grid)
                  guard_v2 = 0
                  while d2 == d1 and guard_v2 < 50:
                      d2 = assign_dose(z_cl[i], test_v2_rho, rng_v2, grid); guard_v2 += 1
                  if d2 == d1:
                      d2 = grid[(grid.index(d1) + 1) % len(grid)]
          else:
              d2 = assign_dose(z_cl[i], rho, rng, grid)
              guard = 0
              while d2 == d1 and guard < 50:
                  d2 = assign_dose(z_cl[i], rho, rng, grid); guard += 1
              if d2 == d1:
                  d2 = grid[(grid.index(d1) + 1) % len(grid)]
          if switch_delay is not None:
              # realistic dose-switch design (see simulate_switch): visit 1 at the exact steady state on d1, visit 2 the
              # interval starting `switch_delay` days after the switch to d2, one continuous trajectory.
              if phase == 'collect':
                  sw_jobs[i] = (p['pk'], d1, d2, p['form'], switch_delay, (p['pid'], float(d1)))
                  continue
              sw = sw_results[i]
              if sw is None:
                  n_no_ss[0 if is_test else 1] += 1
                  continue
              tau = 12.0 if p['form'] == 'Prograf' else 24.0
              if v1_cavg_window is not None and not (v1_cavg_window[0] <= sw['auc1'] / tau <= v1_cavg_window[1]):
                  n_rejected[0 if is_test else 1] += 1
                  continue
              rows += switch_visit_rows(p['pk'], 1, p['pid'] + id_offset, nbr_ss, sw['v1'], sw['auc1'], d1, p['form'], p['ht'], p['cyp'])
              rows += switch_visit_rows(p['pk'], 2, p['pid'] + id_offset, nbr_ss, sw['v2'], sw['auc2'], d2, p['form'], p['ht'], p['cyp'])
          elif v1_cavg_window is not None:
              # clinically plausible exposure on the OBSERVED visit: keep the patient only if the true average
              # concentration over the visit-1 dosing interval (AUC / tau) lies in the window. Doses were already
              # drawn above, so the dose streams are consumed exactly as without the window; the visit-2 dose (the
              # counterfactual) is NOT restricted. Rejected patients are not simulated past visit 1.
              p['pk'].dose_mg = d1
              r1 = generate_patient_visit(p['pk'], 1, p['pid'] + id_offset, nbr_ss, obs, p['form'], p['ht'], p['cyp'])
              tau = 12.0 if p['form'] == 'Prograf' else 24.0
              cavg = float(r1[0]['AUC']) / tau
              if not (v1_cavg_window[0] <= cavg <= v1_cavg_window[1]):
                  n_rejected[0 if is_test else 1] += 1
                  continue
              p['pk'].dose_mg = d2
              if v2_shift is None:
                  rows += r1 + generate_patient_visit(p['pk'], 2, p['pid'] + id_offset, nbr_ss, obs,
                                                      p['form'], p['ht'], p['cyp'])
              else:
                  r2, a0, pv2 = visit2_shifted(p['pk'], v2_shift[0], delta, p['pid'] + id_offset, nbr_ss, obs,
                                               p['form'], p['ht'], p['cyp'])
                  rows += r1 + r2
                  sh = {'v2_shift_param': v2_shift[0], 'v2_shift_delta': delta, f'{v2_shift[0]}_v2': pv2,
                        'AUC_v1': float(r1[0]['AUC']), 'AUC_v2': float(r2[0]['AUC']), 'AUC_v2_noshift': a0}
          else:
            for visit, dose in ((1, d1), (2, d2)):
              p['pk'].dose_mg = dose
              if visit == 2 and v2_shift is not None:
                  r2, a0, pv2 = visit2_shifted(p['pk'], v2_shift[0], delta, p['pid'] + id_offset, nbr_ss, obs,
                                               p['form'], p['ht'], p['cyp'])
                  rows += r2
                  sh = {'v2_shift_param': v2_shift[0], 'v2_shift_delta': delta, f'{v2_shift[0]}_v2': pv2,
                        'AUC_v1': float(r1v[0]['AUC']), 'AUC_v2': float(r2[0]['AUC']), 'AUC_v2_noshift': a0}
              else:
                  r1v = generate_patient_visit(p['pk'], visit, p['pid'] + id_offset, nbr_ss, obs,
                                               p['form'], p['ht'], p['cyp'])
                  rows += r1v
          truth.append(dict(ID=p['pid'] + id_offset, CL_base=p['cl'], confound_par=p['par'],
                            confound_param=confound_param, z_CL=z_cl[i], d1=d1, d2=d2,
                            DRUG=p['form'], CYP=p['cyp'], HT=p['ht'],
                            split='test' if is_test else 'train',
                            **({'switch_delay': switch_delay, 'cap_ratio_d1': float(d1 / (12.0 if p['form'] == 'Prograf' else 24.0) / float(p['pk'].individual_params['Vmax']))} if switch_delay is not None else {}),
                            **({k: float(p['pk'].individual_params[k]) for k in ('Ktr', 'CL', 'Q', 'Vc', 'Vp', 'Vmax', 'Km')
                                if k in p['pk'].individual_params} | {'slow_absorber': p['slow']}
                               | ({'sat_k': p['sat_k']} if sat_on else {})
                               if record_params else {}), **sh))
          if p['pid'] % 50 == 0:
              print(f"  ...{p['pid']}/{num_patients}", flush=True)
    if switch_delay is not None:
        print(f"switch design (delay {switch_delay} d): no visit-1 steady state (dose rate >= capacity) for {n_no_ss[1]} train / {n_no_ss[0]} test patients")
    if v1_cavg_window is not None:
        print(f"visit-1 exposure window {v1_cavg_window} ng/mL: rejected {n_rejected[1]} train / {n_rejected[0]} test patients")
    return pd.DataFrame(rows), pd.DataFrame(truth), n_train


def main():
    ap = argparse.ArgumentParser('Confounded dose assignment (Exp 2)')
    ap.add_argument('--exp', required=True)
    ap.add_argument('--num_patients', type=int, default=1000)
    ap.add_argument('--rho', type=float, default=0.9,
                    help="correlation between standardised log-clearance and the dose "
                         "assignment score. 0 = published independent assignment.")
    ap.add_argument('--scenario', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0, help="physiology stream")
    ap.add_argument('--dose-seed', type=int, default=1234, help="dose-assignment stream")
    ap.add_argument('--test-fraction', type=float, default=0.2)
    ap.add_argument('--randomize-test', action='store_true', default=True)
    ap.add_argument('--no-randomize-test', dest='randomize_test', action='store_false')
    ap.add_argument('--confound-param', default='CL', choices=CONFOUND_PARAMS,
                    help="PK parameter the dose is confounded with. CL sets the curve's "
                         "level and has ZERO leverage on the dose-response; Km sets its "
                         "curvature and is the one that can actually mislead the model.")
    ap.add_argument('--ood-v2-grid', default=None,
                    help="Comma-separated doses for the TEST split's visit-2, drawn from "
                         "outside the training grid (e.g. '0.5,1.0' or '7.0,8.0'). Visit 1 "
                         "and the whole train split keep the normal grid. Default None "
                         "reproduces previous behaviour exactly.")
    ap.add_argument('--test-v2-rho', type=float, default=None,
                    help="Correlation between the confounded parameter and the TEST "
                         "split's visit-2 dose. Default None keeps the randomised "
                         "intervention. Setting it (e.g. -0.9) keeps the same patients "
                         "and visit-1 doses and changes only the counterfactual rule, "
                         "which turns the test split into a probe of whether a model "
                         "is exploiting the training-time dose-physiology association.")
    ap.add_argument('--dose-grid', default=None,
                    help="Comma-separated dose levels, e.g. '1,2,3,4,5,6,7,8'. "
                         "Default is the published 2.0-5.0 grid.")
    ap.add_argument('--zstats-from', default=None,
                    help="Experiment name whose confound_truth.csv supplies the mean and sd "
                         "of log(confound_param) used to standardise z_CL. Use when "
                         "generating a fresh held-out cohort that must sit on the same dose "
                         "scale as the cohort the models were trained on. Default None "
                         "re-derives them from the generated cohort, as before.")
    ap.add_argument('--id-offset', type=int, default=0,
                    help="Added to every patient ID, so a freshly generated cohort can be "
                         "evaluated alongside an existing one without ID collisions.")
    ap.add_argument('--out-root', default='./results/exp_film_run')
    ap.add_argument('--eta-dist', default='log_normal', choices=['log_normal', 'log_t'],
                    help="Distribution of the random effects. log_t = Student-t etas scaled to the same "
                         "variance (heavy tails). Default log_normal = every existing cohort.")
    ap.add_argument('--t-df', type=float, default=4, help="Degrees of freedom for --eta-dist log_t.")
    ap.add_argument('--ktr-slow-frac', type=float, default=0.0,
                    help="Fraction of patients in an unrecorded slow-absorber subgroup (default 0 = none).")
    ap.add_argument('--ktr-slow-factor', type=float, default=1.0,
                    help="Ktr multiplier for the slow-absorber subgroup.")
    ap.add_argument('--ood-keep-v1', action='store_true',
                    help="With --ood-v2-grid: keep every patient's visit-1 dose identical to the in-range cohort "
                         "(the default path does not: its dose stream diverges after the first redraw).")
    ap.add_argument('--v1-cavg-window', default=None,
                    help="'lo,hi' in ng/mL: keep only patients whose true average concentration over the visit-1 "
                         "dosing interval (AUC/tau) lies in [lo, hi]. Default: no window (every existing cohort).")
    ap.add_argument('--prop-sd', type=float, default=None,
                    help="Override gen_tacro_film.RESIDUAL_ERROR_PROP_SD for this run only "
                         "(default: leave the module value, currently 0.03). The paper's "
                         "committed simulator used 0.113.")
    ap.add_argument('--add-sd', type=float, default=None,
                    help="Override gen_tacro_film.RESIDUAL_ERROR_ADD_SD (ng/mL) for this run only "
                         "(default: module value, currently 0.03). The paper used 0.71.")
    ap.add_argument('--sat-scale', type=float, default=1.0,
                    help="Scale every patient's Km AND Vmax by this factor (scenario >= 3). Vmax/Km is unchanged, "
                         "so only the degree of saturation moves (k > 1 = less saturated). Default 1 = no change.")
    ap.add_argument('--sat-logunif', default=None,
                    help="'lo,hi': per-patient Km/Vmax scale drawn log-uniform in [lo, hi] from its own stream "
                         "(a population spread in saturation). Overrides --sat-scale. Default: none.")
    ap.add_argument('--switch-delay', type=float, default=None,
                    help="realistic dose-switch design: visit 1 at the exact steady state on d1, switch to d2 24 h after the "
                         "visit-1 dose, visit 2 = the interval starting DELAY days later (one continuous trajectory). "
                         "Default: none (each visit = 7th dose from an empty body).")
    ap.add_argument('--v2-shift', default=None,
                    help="'PARAM:uniform|normal:SCALE:FRAC', e.g. 'Vmax:uniform:0.7:0.5': at visit 2, PARAM is multiplied "
                         "by exp(delta) for a share FRAC of patients (train and test), delta ~ U(-SCALE, SCALE) or "
                         "N(0, SCALE^2), drawn from its own stream (seed + 6007); the other patients keep their visit-1 "
                         "parameters. Every other stream is consumed as without it, so patients, doses and visit-1 data "
                         "are identical to the cohort generated without the flag. confound_truth.csv gets the delta, the "
                         "visit-2 value, AUC_v1 / AUC_v2 and AUC_v2_noshift (visit-2 dose, visit-1 parameters). "
                         "Default: none (both visits share every parameter, as in every earlier cohort).")
    a = ap.parse_args()
    v2_shift = None
    if a.v2_shift:
        _p, _d, _s, _f = a.v2_shift.split(':')
        assert _p in ('Ktr', 'Q', 'Vc', 'Vp', 'Vmax', 'Km') and _d in ('uniform', 'normal'), a.v2_shift
        v2_shift = (_p, _d, float(_s), float(_f))
        print(f"visit-2 change: {_p} x exp(delta), delta ~ {_d}({float(_s):g}) for {100 * float(_f):g} % of patients")
    import gen_tacro_film as _gtf
    if a.prop_sd is not None: _gtf.RESIDUAL_ERROR_PROP_SD = a.prop_sd
    if a.add_sd is not None: _gtf.RESIDUAL_ERROR_ADD_SD = a.add_sd
    print(f"residual error: sd = {_gtf.RESIDUAL_ERROR_ADD_SD} + {_gtf.RESIDUAL_ERROR_PROP_SD} * C")

    out = os.path.join(a.out_root, str(a.exp)); os.makedirs(out, exist_ok=True)
    grid = [float(x) for x in a.dose_grid.split(',')] if a.dose_grid else None
    ood = [float(x) for x in a.ood_v2_grid.split(',')] if a.ood_v2_grid else None
    zstats = None
    if a.zstats_from:
        _t = pd.read_csv(os.path.join(a.out_root, a.zstats_from, 'confound_truth.csv'))
        _lp = np.log(_t['confound_par'].values)
        zstats = (_lp.mean(), _lp.std())
        print(f"z_CL standardisation pinned to {a.zstats_from}: "
              f"mean={zstats[0]:.6f} sd={zstats[1]:.6f}")
    df, truth, n_train = build(a.num_patients, a.rho, a.scenario, a.seed, a.dose_seed,
                               a.test_fraction, a.randomize_test, ood_v2_grid=ood,
                               zstats=zstats, id_offset=a.id_offset,
                               test_v2_rho=a.test_v2_rho,
                               confound_param=a.confound_param, grid=grid,
                               eta_dist=a.eta_dist, t_df=a.t_df,
                               ktr_slow_frac=a.ktr_slow_frac, ktr_slow_factor=a.ktr_slow_factor,
                               v1_cavg_window=([float(x) for x in a.v1_cavg_window.split(',')]
                                               if a.v1_cavg_window else None),
                               ood_keep_v1=a.ood_keep_v1, sat_scale=a.sat_scale,
                               sat_logunif=([float(x) for x in a.sat_logunif.split(',')]
                                            if a.sat_logunif else None),
                               switch_delay=a.switch_delay, v2_shift=v2_shift,
                               switch_cache=(os.path.join(a.out_root, f'_switch_cache_seed{a.seed}_sat{a.sat_scale:g}_{a.scenario}.pkl')
                                             if a.switch_delay is not None else None))
    tr = truth[truth.split == 'train'].ID.values
    te = truth[truth.split == 'test'].ID.values
    df[df['ID'].isin(tr)].to_csv(os.path.join(out, 'virtual_cohort_film_train.csv'), index=False)
    df[df['ID'].isin(te)].to_csv(os.path.join(out, 'virtual_cohort_film_test.csv'), index=False)
    truth.to_csv(os.path.join(out, 'confound_truth.csv'), index=False)
    import json as _json
    with open(os.path.join(out, 'noise.json'), 'w') as _f:
        _json.dump({'add_sd': _gtf.RESIDUAL_ERROR_ADD_SD, 'prop_sd': _gtf.RESIDUAL_ERROR_PROP_SD,
                    'v1_cavg_window': a.v1_cavg_window,
                    **({'switch_delay': a.switch_delay} if a.switch_delay is not None else {}),
                    **({'v2_shift': a.v2_shift} if a.v2_shift else {}),
                    **({'sat_scale': a.sat_scale, 'sat_logunif': a.sat_logunif}
                       if (a.sat_scale != 1.0 or a.sat_logunif) else {})}, _f)

    trn = truth[truth.split == 'train']; tst = truth[truth.split == 'test']
    print(f"\nWrote {out}\n  train {len(tr)} / test {len(te)} | rho = {a.rho} on {a.confound_param}")
    print(f"  dose grid: {grid or GRID}  ({(grid or GRID)[-1]/(grid or GRID)[0]:.1f}x span)")
    print(f"  realised corr(log {a.confound_param}, d1): train "
          f"{np.corrcoef(np.log(trn.confound_par), trn.d1)[0,1]:+.3f} | test "
          f"{np.corrcoef(np.log(tst.confound_par), tst.d1)[0,1]:+.3f}")
    print(f"  realised corr(log CL, d1): train {np.corrcoef(np.log(trn.CL_base), trn.d1)[0,1]:+.3f}"
          f" | test {np.corrcoef(np.log(tst.CL_base), tst.d1)[0,1]:+.3f}")
    print(f"  realised corr(log CL, d2): train {np.corrcoef(np.log(trn.CL_base), trn.d2)[0,1]:+.3f}"
          f" | test {np.corrcoef(np.log(tst.CL_base), tst.d2)[0,1]:+.3f}"
          f"   <- test d2 is the randomised intervention" if a.randomize_test else "")


if __name__ == '__main__':
    main()
