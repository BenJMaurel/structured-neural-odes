#!/usr/bin/env python3
"""Part 2 design check (2026-10-03): does a CONTINUOUS per-patient timeline (visit 1 at d1, then d2 for NBR_SS more
intervals, then the observed visit-2 interval) give the same visit-2 curve as the cohort's FRESH steady state at d2?
If yes, the EB-VAE can be trained/scored on the cohort's own observations (identical data to our model).
True parameters from the replay (exact for confound_vc00_s4_pnoise_win). Random sample of test patients, in/lo/hi d2."""
import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, 'scripts/ident'); sys.path.insert(0, '.')
from ebe_s4_batch import simulate, dosing, FIT
from oracle_floor import BatchPK, NBR_SS
from mle_km_s4 import replay
REL = np.array([0., 0.33, 0.67, 1., 1.5, 2., 3., 4., 6., 9., 12., 24.])
class SeqPK(BatchPK):                       # per-event dose amounts (dose change mid-timeline)
    def state_update(self, state):
        A = list(state); A[0] = A[0] + self.seq[self.k]; self.k += 1; return tuple(A)
def continuous(theta, d1, d2, form):
    M = theta.shape[0]; tau = 12.0 if form == 'Prograf' else 24.0
    dos1 = dosing(form); t_end1 = 24.0 * NBR_SS
    dos2 = [t_end1 + tau * j for j in range(1, NBR_SS + 2)]          # NBR_SS+1 doses of d2, observe after the last
    t_obs = dos2[-1] + REL[REL <= tau]; hi = tau; ta = dos2[-1] + np.round(np.arange(0, hi + 1e-9, 0.02), 4)
    pk = SeqPK(M, formulation=form, hematocrit=35.0, cyp_status='non_expresser', scenario=4)
    pk.individual_params = {n: torch.as_tensor(theta[:, i], dtype=torch.float32) for i, n in enumerate(FIT)}
    pk.individual_params['CL'] = torch.full((M,), 20.0)
    pk.seq = [torch.as_tensor(d1, dtype=torch.float32)] * len(dos1) + [torch.as_tensor(d2, dtype=torch.float32)] * len(dos2); pk.k = 0
    dos = dos1 + dos2
    tp = torch.unique(torch.cat([torch.arange(1.0, dos[-1], 1.0), torch.tensor(dos[1:]), torch.tensor(ta, dtype=torch.float32)]))
    with torch.no_grad(): c = pk.simulate(dos, tp[tp > 0], resample=False) * 1000.0
    tpp = tp[tp > 0]; idx = [int(torch.argmin(torch.abs(tpp - t))) + 1 for t in torch.tensor(ta, dtype=torch.float32)]
    return np.trapezoid(c[idx, :].double().numpy(), ta - ta[0], axis=0)
C = 'confound_vc00_s4_pnoise_win'; T = pd.read_csv(f'results/exp_film_run/{C}/confound_truth.csv').set_index('ID')
P = replay(int(T.index.max()), 4)
te = T[T.split == 'test'].index.values; rng = np.random.RandomState(0); ids = rng.choice(te, 120, replace=False)
for reg, coh in (('in', C), ('lo', C + '_lo'), ('hi', C + '_hi')):
    Tr = pd.read_csv(f'results/exp_film_run/{coh}/confound_truth.csv').set_index('ID'); out = []
    for form in ('Advagraf', 'Prograf'):
        g = [i for i in ids if P.loc[i].form == form]
        if not g: continue
        th = np.stack([[P.loc[i][n] for n in FIT] for i in g]); d1 = Tr.loc[g].d1.values.astype(float); d2 = Tr.loc[g].d2.values.astype(float)
        tau = 12.0 if form == 'Prograf' else 24.0; ta = np.round(np.arange(0, tau + 1e-9, 0.02), 4)
        fresh = np.trapezoid(simulate(th, np.full(len(g), 20.0), d2, form, ta), ta, axis=0)
        cont = continuous(th, d1, d2, form)
        out += list(cont / fresh - 1)
    r = np.abs(np.array(out)) * 100
    print(f'{reg}: n={len(out)}  |AUC continuous / fresh steady state - 1|: median {np.median(r):.3f}%  p95 {np.percentile(r, 95):.3f}%  max {r.max():.3f}%')
