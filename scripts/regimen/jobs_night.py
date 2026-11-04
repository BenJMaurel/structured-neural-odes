#!/usr/bin/env python3
"""Job lists for scripts/regimen/queue_night.sh. One TSV line per job: out \t log \t workdir \t env \t command.
A job is skipped when its output exists. List 1 (priority order): B2 encoder diagnosis, improved-B penalty sweep (linear and
saturable simulator), saturable-simulator baselines, remaining fairness checks, EB-VAE curves. List 2 (after the barrier):
improved B with titrated arms at the penalty chosen on the VALIDATION cohort (never on the test), and at penalty 0.
    jobs_night.py list1 > file   |   jobs_night.py list2 > file"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, glob, numpy as np
ROOT = REPO; EB = f'{REPO}/third_party/baaz_ebvae'
R = f'{ROOT}/results/regimen_dh'; PY = 'python3'; LOGS = f'{ROOT}/logs/regimen_dh'
KM = '500'
# ONE SEED PER TEST :
# sparse seed 1 (a typical draw; seed 0's randomised arms correlate +0.27 with log ke), dense seed 0 (the only dense seed with
# linear-simulator baselines already run; it shares seed 0's patients and imbalance).
SPARSE = [('sparse', 1)]; DENSE = [('dense', 0)]
LAMS_SPARSE = ('0', '100', '1000', '10000'); LAMS_DENSE = ('0', '1000', '10000')
SIMS = (('lin', ''), (f'mm{KM}', f'REGIMEN_KM={KM}'))
out = []


def job(o, log, cmd, wd=ROOT, env=''):
    out.append('\t'.join([o, f'{LOGS}/{log}.log', wd, env or '-', cmd]))


def v3(sim, env, lam, d, s, titr=False, extra=''):
    sub = f'{R}/v3/{sim}/' + ('titr0.8/' if titr else '') + f'lam{lam}'
    o = f'{sub}/N100_{d}_s{s}.json'
    job(o, f'v3_{sim}{"_titr" if titr else ""}_lam{lam}_{d}_s{s}',
        f'{PY} scripts/regimen/dh_regimen_v3.py 100 {d} {s} {o} --iters 6000 --resid-dyn {lam} --resid-read {lam}'
        + (' --titrate 0.8' if titr else '') + extra, env=env)


def list1():
    # 1. B2 encoder diagnosis (iScience format) + MAP-BE oracle there
    for d, s in SPARSE + DENSE:
        o = f'{R}/firstint/mapbe_oracle/N100_{d}_s{s}.json'
        job(o, f'b2diag_mapbe_N100_{d}_s{s}', f'{PY} scripts/regimen/mapbe_regimen.py 100 {d} {s} {o} firstint')
    for d, s in SPARSE + DENSE:
        for e in ('norm', 'none'):
            o = f'{R}/firstint/B2_enc{e}/N100_{d}_s{s}.json'
            job(o, f'b2diag_{e}_N100_{d}_s{s}', f'{PY} scripts/regimen/dh_regimen_v2.py 100 {d} {s} {o} --iters 6000 --protocol firstint --enc-dose {e}')
    # 2. improved B, penalty sweep, sparse (randomised arms), both simulators
    for lam in LAMS_SPARSE:
        for sim, env in SIMS:
            for d, s in SPARSE:
                v3(sim, env, lam, d, s)
    # 3. saturable-simulator baselines, sparse: B2, B, EB-VAE (randomised + titrated), MAP-BE oracle
    env = f'REGIMEN_KM={KM}'; M = f'{R}/mm{KM}'
    for d, s in SPARSE:
        o = f'{M}/mapbe_oracle/N100_{d}_s{s}.json'; job(o, f'mm_mapbe_{d}_s{s}', f'{PY} scripts/regimen/mapbe_regimen.py 100 {d} {s} {o}', env=env)
        for titr in (False, True):
            t = ' --titrate 0.8' if titr else ''; sub = 'titr0.8/' if titr else ''; tag = '_titr' if titr else ''
            o = f'{M}/{sub}B2/N100_{d}_s{s}.json'; job(o, f'mm_B2{tag}_{d}_s{s}', f'{PY} scripts/regimen/dh_regimen_v2.py 100 {d} {s} {o} --iters 6000 --no-check{t}', env=env)
            o = f'{M}/{sub}B/N100_{d}_s{s}.json'; job(o, f'mm_B{tag}_{d}_s{s}', f'{PY} scripts/regimen/dh_regimen.py 100 {d} {s} {o} --iters 6000 --no-check{t}', env=env)
            o = f'{EB}/ours/regimen/res_mm{KM}/{"titr0.8" if titr else "ctrl"}/N100_{d}_s{s}.json'
            job(o, f'mm_ebvae{tag}_{d}_s{s}', f'{PY} -u ours/regimen/run_regimen_titrated.py 100 {d} {s} {"0.8" if titr else "none"} {o}',
                wd=EB, env=f'{env} MKL_NUM_THREADS=1 MPLBACKEND=Agg')
    # 4. ablation: B2 with learned noise (improved B with every correction off), sparse, both simulators
    for sim, env in SIMS:
        for d, s in SPARSE:
            o = f'{R}/v3/{sim}/noise_learned_nocorr/N100_{d}_s{s}.json'
            job(o, f'v3_{sim}_noiselearned_{d}_s{s}', f'{PY} scripts/regimen/dh_regimen_v3.py 100 {d} {s} {o} --iters 6000 --noise learned', env=env)
    # 5. dense: improved B sweep + saturable baselines
    for lam in LAMS_DENSE:
        for sim, env in SIMS:
            for d, s in DENSE:
                v3(sim, env, lam, d, s)
    for d, s in DENSE:
        o = f'{M}/mapbe_oracle/N100_{d}_s{s}.json'; job(o, f'mm_mapbe_{d}_s{s}', f'{PY} scripts/regimen/mapbe_regimen.py 100 {d} {s} {o}', env=env)
        for titr in (False, True):
            t = ' --titrate 0.8' if titr else ''; sub = 'titr0.8/' if titr else ''; tag = '_titr' if titr else ''
            o = f'{M}/{sub}B2/N100_{d}_s{s}.json'; job(o, f'mm_B2{tag}_{d}_s{s}', f'{PY} scripts/regimen/dh_regimen_v2.py 100 {d} {s} {o} --iters 6000 --no-check{t}', env=env)
            o = f'{M}/{sub}B/N100_{d}_s{s}.json'; job(o, f'mm_B{tag}_{d}_s{s}', f'{PY} scripts/regimen/dh_regimen.py 100 {d} {s} {o} --iters 6000 --no-check{t}', env=env)
            o = f'{EB}/ours/regimen/res_mm{KM}/{"titr0.8" if titr else "ctrl"}/N100_{d}_s{s}.json'
            job(o, f'mm_ebvae{tag}_{d}_s{s}', f'{PY} -u ours/regimen/run_regimen_titrated.py 100 {d} {s} {"0.8" if titr else "none"} {o}',
                wd=EB, env=f'{env} MKL_NUM_THREADS=1 MPLBACKEND=Agg')
    # 6. remaining fairness checks: Lu models in the iScience format (the tuning sweep's test seed, sparse 0, is complete)
    for d, s in SPARSE + DENSE:
        for m in ('is', 'nmi'):
            o = f'{R}/firstint/lu{m}/N100_{d}_s{s}.json'
            job(o, f'fair_fi_lu{m}_N100_{d}_s{s}', f'{PY} scripts/regimen/lu_regimen.py {m} 100 {d} {s} {o} --iters 6000 --protocol firstint')
    # 7. EB-VAE curves for the patient-level Test 2 figure
    for rho in ('none', '0.8'):
        o = f'{EB}/ours/regimen/res_curves/N100_sparse_s1_{rho}.json'
        job(o, f'ebcurves_{rho}', f'{PY} -u ours/regimen/run_regimen_titrated.py 100 sparse 1 {rho} {o}', wd=EB, env='DUMP_CURVES=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg')


def selected_lambda(sim, design):
    """penalty with the lowest mean VALIDATION RMSE (fresh patients, training regimen) over the seeds of this design"""
    best = None
    for lam in (LAMS_SPARSE if design == 'sparse' else LAMS_DENSE):
        fs = glob.glob(f'{R}/v3/{sim}/lam{lam}/N100_{design}_s*.json')
        if not fs:
            continue
        v = np.mean([json.load(open(f))['validation']['rmse_mgL'] for f in fs])
        if best is None or v < best[1]:
            best = (lam, v)
    return best[0] if best else None


def list2():
    sel = {}
    for sim, env in SIMS:
        for design, cfgs in (('sparse', SPARSE), ('dense', DENSE)):
            lam = selected_lambda(sim, design); sel[f'{sim}_{design}'] = lam
            for l in sorted({lam, '0'} - {None}):
                for d, s in cfgs:
                    v3(sim, env, l, d, s, titr=True)
    os.makedirs(f'{R}/v3', exist_ok=True)
    json.dump(sel, open(f'{R}/v3/selected_lambda.json', 'w'), indent=1)


if __name__ == '__main__':
    {'list1': list1, 'list2': list2}[sys.argv[1]]()
    print('\n'.join(out))
