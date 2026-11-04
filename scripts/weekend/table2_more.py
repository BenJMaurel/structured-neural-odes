#!/usr/bin/env python3
"""Table 2 with ten seeds. OT-FiLM residual vs dose-cond residual on the windowed two-visit task,
100 training patients, ep6000. Seeds 1-3: results/s4_pnoise_win/eval (the published runs); seeds 4-10: results/s4_pnoise_win_more/eval
(scripts/weekend/queue_weekend.sh, same commands). Cells, aggregation and the paired patient bootstrap are those of
scripts/ndm/ndm_tables.py (imported, unchanged), so the numbers are computed exactly as the published ones.
CHECK: the seed-1 rerun (results/s4_pnoise_win_more/eval/..._s1_...) against the published seed 1.
PRE-STATED: repro within 0.3 points at V1 and V2; OT-FiLM lower than dose-cond in >= 8/10 seed pairs at V1 and at V2; patient-bootstrap
interval of the difference excludes 0 at both visits.
Writes results/s4_pnoise_win_more/table2_more.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys
import numpy as np
os.chdir(REPO); sys.path.insert(0, 'scripts/ndm')
import ndm_tables as T

OLD, NEW = 'results/s4_pnoise_win/eval', 'results/s4_pnoise_win_more/eval'
SEEDS = list(range(1, 11))
path = lambda root, arch, s, reg: f'{root}/ep006000_{arch}_n100_s{s}_on_{reg}.json'
src = lambda arch, s, reg: path(OLD if s <= 3 else NEW, arch, s, reg)
L = [__doc__.split('\n')[0], '']

L.append('reproducibility (seed 1 rerun with the current code vs the published seed 1), median |AUC err| %:')
for arch in ('film', 'dc'):
    for visit in ('v1', 'v2'):
        a, b = T.net_cell(path(OLD, arch, 1, 'in'), visit), T.net_cell(path(NEW, arch, 1, 'in'), visit)
        if a is None or b is None:
            L.append(f'  {arch} {visit}: not available'); continue
        d = np.abs(a['abs'] - b['abs']).max()
        L.append(f'  {arch:4s} {visit}: published {a["med"]:.2f}  rerun {b["med"]:.2f}  (max per-patient |diff| {100 * d:.3g} pt)')
L.append('')

for visit, reg, title in T.REGS:
    cells = {arch: [T.net_cell(src(arch, s, reg), visit) for s in SEEDS] for arch in ('film', 'dc')}
    got = {arch: [s for s, c in zip(SEEDS, cells[arch]) if c is not None] for arch in cells}
    common = sorted(set(got['film']) & set(got['dc']))
    A = {arch: T.agg([cells[arch][s - 1] for s in common]) for arch in cells}
    L.append(f'== {title}  (seeds {common})')
    L.append(f'  OT-FiLM         {T.fmt(A["film"])}')
    L.append(f'  dose-cond field {T.fmt(A["dc"])}')
    L.append(T.paired(A['film'], A['dc'], 'OT-FiLM - dose-cond'))
    if A['film'] is not None and A['dc'] is not None:
        d = A['film']['med_seeds'] - A['dc']['med_seeds']
        L.append(f'  OT-FiLM lower in {int((d < 0).sum())}/{len(d)} seed pairs')
    L.append('')
os.makedirs('results/s4_pnoise_win_more', exist_ok=True)
open('results/s4_pnoise_win_more/table2_more.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
