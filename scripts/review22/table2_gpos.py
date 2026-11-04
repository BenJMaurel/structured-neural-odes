#!/usr/bin/env python3
"""Table 2 with gamma > 0 (review 22/10): OT-FiLM retrained with gamma = softplus(.) (--film-gamma-pos, results/s4_pnoise_win_gpos),
ten seeds, same cohort, flags, budget and scoring as the published OT-FiLM row (ep6000, 1100 in-range test patients, scaler pinned
to the training cohort). Compared with the dose-conditioned field (Table 2's ten seeds) and with the published OT-FiLM seeds, with the
cells, aggregation and paired patient bootstrap of scripts/ndm/ndm_tables.py (as scripts/weekend/table2_more.py).
Outlier screen as table2_outliers.py: a seed is flagged when its median error is > 3 x 1.4826 MAD from the seed median.
Writes results/review22/table2_gpos.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys
import numpy as np
os.chdir(REPO); sys.path.insert(0, 'scripts/ndm')
import ndm_tables as T

SEEDS = list(range(1, 11))
P = {'gpos': lambda s: f'results/s4_pnoise_win_gpos/eval/ep006000_film_n100_s{s}_on_in.json',
     'film': lambda s: f'results/{"s4_pnoise_win" if s <= 3 else "s4_pnoise_win_more"}/eval/ep006000_film_n100_s{s}_on_in.json',
     'dc': lambda s: f'results/{"s4_pnoise_win" if s <= 3 else "s4_pnoise_win_more"}/eval/ep006000_dc_n100_s{s}_on_in.json'}
NAME = {'gpos': 'OT-FiLM, gamma > 0', 'film': 'OT-FiLM (published)', 'dc': 'dose-cond field'}
L = [__doc__.split('\n')[0], '']
A = {}
for visit in ('v1', 'v2'):
    cells = {k: [T.net_cell(p(s), visit) for s in SEEDS] for k, p in P.items()}
    common = [s for s in SEEDS if all(cells[k][s - 1] is not None for k in P)]
    A[visit] = {k: T.agg([cells[k][s - 1] for s in common]) for k in P}
    L.append(f'== {visit}  (seeds {common})')
    for k in P:
        L.append(f'  {NAME[k]:20s} {T.fmt(A[visit][k])}')
    for a, b in (('gpos', 'dc'), ('gpos', 'film')):
        L.append(T.paired(A[visit][a], A[visit][b], f'{NAME[a]} - {NAME[b]}'))
        d = A[visit][a]['med_seeds'] - A[visit][b]['med_seeds']
        L.append(f'  {NAME[a]} lower than {NAME[b]} in {int((d < 0).sum())}/{len(d)} seed pairs')
    m = A[visit]['gpos']['med_seeds']; mad = 1.4826 * np.median(np.abs(m - np.median(m)))
    out = [s for s, x in zip(common, m) if abs(x - np.median(m)) > 3 * mad]
    L.append(f'  outlier seeds (gamma > 0): {out or "none"}  (median {np.median(m):.1f}, 3 x MAD {3 * mad:.1f})')
    L.append('')
g1, g2 = A['v1']['gpos'], A['v2']['gpos']
L.append('Table 2 row (LaTeX):')
L.append(f'OT-FiLM & {g1["med"]:.1f} ({g1["med_sd"]:.1f}) & {g1["w20"]:.1f} & {g2["med"]:.1f} ({g2["med_sd"]:.1f}) & {g2["w20"]:.1f} '
         f'& {g2["trm"]:.1f} & {g2["n100"]:.1f} \\\\')
open('results/review22/table2_gpos.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
