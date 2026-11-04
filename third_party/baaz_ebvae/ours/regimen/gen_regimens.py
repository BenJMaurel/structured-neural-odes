"""Their simulated PK (lib/utils/utils_data_generation.simulate_single_drug_concentration), vectorised over patients so
the SAME patients can be simulated under several regimens (individual counterfactual truth). Identical model: correlated
log-normal (ka, ke, v) with their means/sds/correlation, A_central starts at v (their 'C0'), doses into the gut at t >=
dose time, Euler dt = 0.01 with clipping at 0, concentration = A_central, noise C*(1+N(0,prop)) + N(0,add), clipped >= 0.
    gen_regimens.py <n_patients> <seed> <out_prefix>   -> <prefix>_params.csv and true curves on the 0.01-h grid (npz)"""
import os, sys, numpy as np, pandas as pd
# Saturable (Michaelis-Menten) elimination, opt-in (PharmaNODE 2026-10-06): REGIMEN_KM=<Km> in the environment
# replaces the linear elimination ke * A_c by Vmax * A_c / (Km + A_c) with Vmax = ke * Km, so the low-concentration behaviour
# of every patient is unchanged; unset -> their linear model, byte-identical.
KM = float(os.environ['REGIMEN_KM']) if os.environ.get('REGIMEN_KM') else None
KA, KE, V, SD = 0.4, 0.6, 50.0, 0.5
CORR = np.array([[1.0, 0.3, 0.35], [0.3, 1.0, 0.2], [0.35, 0.2, 1.0]])
ADD_E, PROP_E, DT, T_END = 10.0, 0.0001, 0.01, 30.0
REGIMENS = {   # name: (times, amounts)  -- amounts None = the patient's training arm (400 or 800)
    'train':  ([0, 8, 16, 24], None),
    'amt350': ([0, 8, 16, 24], [350] * 4), 'amt600': ([0, 8, 16, 24], [600] * 4), 'amt850': ([0, 8, 16, 24], [850] * 4),
    'time400': ([0, 3, 8], [400] * 3), 'time800': ([0, 3, 8], [800] * 3),
}
# Held-out confirmation regimens, opt-in (PharmaNODE 2026-10-17): REGIMENS_CONFIRM=1 appends five regimens fixed
# before any confirmation run and never used for a design choice. Their dose times are restricted to those above (0, 3, 8, 16, 24 h)
# because run_regimen_titrated.py builds the EB-VAE's training time grid from the dose times of every regimen: training is unchanged.
# Unset -> the dict above, byte-identical.
if os.environ.get('REGIMENS_CONFIRM'):
    REGIMENS.update({
        'c_amt500': ([0, 8, 16, 24], [500] * 4),             # between the two training doses
        'c_amt1000': ([0, 8, 16, 24], [1000] * 4),           # above the highest training dose
        'c_load': ([0, 8, 16, 24], [1200, 400, 400, 400]),   # loading dose, then 400 mg
        'c_missed': ([0, 8, 24], [800] * 3),                 # 800 mg with the 16-h dose missed
        'c_extra': ([0, 3, 8, 16, 24], [400] * 5),           # 400 mg with an extra dose at 3 h
    })
def draw(n, rng):
    cov = np.outer([SD] * 3, [SD] * 3) * CORR
    return np.exp(rng.multivariate_normal(np.log([KA, KE, V]), cov, size=n))      # columns ka, ke, v
def simulate(params, times, amounts):
    """params [n,3]; times list; amounts [n, n_doses] -> concentrations on t_eval [n, T]"""
    ka, ke, v = params[:, 0], params[:, 1], params[:, 2]; n = len(ka)
    t_eval = np.arange(0.0, T_END + DT / 2, DT); A_gut = np.zeros(n); A_c = v.copy(); out = np.zeros((n, len(t_eval))); k = 0
    for i, t in enumerate(t_eval):
        while k < len(times) and t >= times[k]:
            A_gut = A_gut + amounts[:, k]; k += 1
        out[:, i] = A_c
        dg = -ka * A_gut; dc = ka * A_gut - (ke * A_c if KM is None else ke * KM * A_c / (KM + A_c))
        A_gut = np.maximum(A_gut + dg * DT, 0); A_c = np.maximum(A_c + dc * DT, 0)
    return t_eval, out
def noisy(c, rng):
    return np.maximum(0, c * (1 + rng.normal(0, PROP_E, c.shape)) + rng.normal(0, ADD_E, c.shape))
def titrate(P, rho, seed):
    """Titrated training arms (PharmaNODE 2026-10-05): fast eliminators get the high dose. score =
    rho * z(log ke) + sqrt(1 - rho^2) * N(0, 1), drawn from its OWN stream (seed + 9001) so the caller's data stream --
    hence every patient and noise draw -- is identical to the randomised design; the top half of the score gets 800 mg,
    the rest 400 mg (exactly balanced, as the alternating arms). Returns (arm, corr(log ke, arm == 800))."""
    z = (np.log(P[:, 1]) - np.log(KE)) / SD
    score = rho * z + np.sqrt(1 - rho ** 2) * np.random.default_rng(seed + 9001).standard_normal(len(P))
    arm = np.where(score > np.median(score), 800.0, 400.0)
    return arm, float(np.corrcoef(np.log(P[:, 1]), arm == 800.0)[0, 1])
if __name__ == '__main__':
    n, seed, pref = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rng = np.random.default_rng(seed); P = draw(n, rng); arm = np.where(np.arange(n) % 2 == 0, 400.0, 800.0)
    curves = {}
    for name, (times, amts) in REGIMENS.items():
        A = np.repeat(arm[:, None], len(times), 1) if amts is None else np.tile(np.array(amts, float), (n, 1))
        t_eval, curves[name] = simulate(P, times, A)
    pd.DataFrame(dict(ID=np.arange(1, n + 1), ka=P[:, 0], ke=P[:, 1], v=P[:, 2], arm=arm)).to_csv(f'{pref}_params.csv', index=False)
    np.savez_compressed(f'{pref}_curves.npz', t=t_eval, **curves)
    print(f'{n} patients, regimens {list(curves)}, curves on {len(t_eval)} points')
