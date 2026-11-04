import sys, numpy as np, pandas as pd, torch
sys.path.insert(0, 'scripts/ident'); sys.path.insert(0, '.')
import oracle_floor, ebe_s4_batch
from ebe_s4_batch import FIT
from oracle_floor import BatchPK
from mle_km_s4 import replay
def auc_after(theta, dose, form, n_doses):
    M = theta.shape[0]; tau = 12.0 if form == 'Prograf' else 24.0
    dos = [tau * j for j in range(n_doses)]; ta = dos[-1] + np.round(np.arange(0, tau + 1e-9, 0.05), 4)
    pk = BatchPK(M, formulation=form, hematocrit=35.0, cyp_status='non_expresser', scenario=4)
    pk.individual_params = {n: torch.as_tensor(theta[:, i], dtype=torch.float32) for i, n in enumerate(FIT)}
    pk.individual_params['CL'] = torch.full((M,), 20.0); pk.dose_mg = torch.as_tensor(dose, dtype=torch.float32)
    tp = torch.unique(torch.cat([torch.arange(1.0, dos[-1], 2.0), torch.tensor(dos[1:]), torch.tensor(ta, dtype=torch.float32)]))
    tp = tp[tp > 0]
    with torch.no_grad(): c = pk.simulate(dos, tp, resample=False) * 1000.0
    idx = [int(torch.argmin(torch.abs(tp - t))) + 1 for t in torch.tensor(ta, dtype=torch.float32)]
    return np.trapezoid(c[idx, :].double().numpy(), ta - ta[0], axis=0)
C = 'confound_vc00_s4_pnoise_win'; T = pd.read_csv(f'results/exp_film_run/{C}/confound_truth.csv').set_index('ID')
P = replay(int(T.index.max()), 4); te = T[T.split == 'test'].index.values; ids = np.random.RandomState(0).choice(te, 120, replace=False)
print(f'generator: NBR_SS = {oracle_floor.NBR_SS} previous doses, observed after dose {oracle_floor.NBR_SS + 1}')
for lab, col in (('V1 dose d1', 'd1'), ('V2 dose d2 (in range)', 'd2')):
    r = []
    for form in ('Advagraf', 'Prograf'):
        g = [i for i in ids if P.loc[i].form == form]; th = np.stack([[P.loc[i][n] for n in FIT] for i in g]); d = T.loc[g][col].values.astype(float)
        a7 = auc_after(th, d, form, oracle_floor.NBR_SS + 1); a40 = auc_after(th, d, form, 40)
        r += list(a7 / a40 - 1)
    r = 100 * np.array(r)
    print(f'{lab}: AUC after 7 doses / after 40 doses - 1: median {np.median(r):+.1f}%  p5 {np.percentile(r, 5):+.1f}%  p95 {np.percentile(r, 95):+.1f}%  min {r.min():+.1f}%  (share within 5 % of steady state: {np.mean(np.abs(r) < 5) * 100:.0f}%)')

class SeqPK(BatchPK):
    def state_update(self, state):
        A = list(state); A[0] = A[0] + self.seq[self.k]; self.k += 1; return tuple(A)
def washout_auc(theta, d1, d2, form, W):
    M = theta.shape[0]; tau = 12.0 if form == 'Prograf' else 24.0
    dos1 = [tau * j for j in range(7)]; start2 = dos1[-1] + tau + W; dos2 = [start2 + tau * j for j in range(7)]
    ta = dos2[-1] + np.round(np.arange(0, tau + 1e-9, 0.05), 4)
    pk = SeqPK(M, formulation=form, hematocrit=35.0, cyp_status='non_expresser', scenario=4)
    pk.individual_params = {n: torch.as_tensor(theta[:, i], dtype=torch.float32) for i, n in enumerate(FIT)}
    pk.individual_params['CL'] = torch.full((M,), 20.0)
    pk.seq = [torch.as_tensor(d1, dtype=torch.float32)] * 7 + [torch.as_tensor(d2, dtype=torch.float32)] * 7; pk.k = 0
    dos = dos1 + dos2
    tp = torch.unique(torch.cat([torch.arange(1.0, dos[-1], 2.0), torch.tensor(dos[1:]), torch.tensor(ta, dtype=torch.float32)])); tp = tp[tp > 0]
    with torch.no_grad(): c = pk.simulate(dos, tp, resample=False) * 1000.0
    idx = [int(torch.argmin(torch.abs(tp - t))) + 1 for t in torch.tensor(ta, dtype=torch.float32)]
    return np.trapezoid(c[idx, :].double().numpy(), ta - ta[0], axis=0)
for W in (96, 192, 336, 504):
    r = []
    for form in ('Advagraf', 'Prograf'):
        g = [i for i in ids if P.loc[i].form == form]; th = np.stack([[P.loc[i][n] for n in FIT] for i in g])
        d1 = T.loc[g].d1.values.astype(float); d2 = T.loc[g].d2.values.astype(float)
        r += list(washout_auc(th, d1, d2, form, W) / auc_after(th, d2, form, 7) - 1)
    r = 100 * np.abs(np.array(r))
    print(f'washout {W:3d} h: visit-2 AUC vs fresh 7-dose course: median {np.median(r):.4f}%  p95 {np.percentile(r, 95):.4f}%  max {r.max():.4f}%')
