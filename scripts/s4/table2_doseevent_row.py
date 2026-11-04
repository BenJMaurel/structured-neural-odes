#!/usr/bin/env python3
"""Main-text Table 2, dose-event row (fallback (a)): dh7pgkp = jump b(p), encoder reads the samples and the dose,
windowed scenario 4, N=100, ep6000, 1100 in-range test patients. Cells with the definitions of scripts/ndm/ndm_tables.py (imported:
net_cell / agg): V1 med (sd over seeds), V1 +-20 %, V2 med (sd), V2 +-20 %, V2 RMSPE after dropping the worst 1 %, V2 count > 100 %,
means over seeds. dh7pg (the previous row, 12.7 (0.6) / 15.5 (0.4)) is printed as a check of the definitions.
Writes results/s4_doseh/table2_doseevent_row.txt"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ndm'))
from ndm_tables import net_cell, agg

E = 'results/s4_doseh/eval'
out = [__doc__.split('\n')[0], '']
for v in ('dh7pg', 'dh7pgkp'):
    a = {vis: agg([net_cell(f'{E}/ep006000_{v}_osplit1-5_n100_s{s}_on_in.json', vis) for s in (1, 2, 3)]) for vis in ('v1', 'v2')}
    if a['v1'] is None:
        out.append(f'{v}: not evaluated'); continue
    v1, v2 = a['v1'], a['v2']
    out.append(f"{v:8s} n={v1['n']}  V1 {v1['med']:.1f} ({v1['med_sd']:.1f}) & {v1['w20']:.1f} & V2 {v2['med']:.1f} ({v2['med_sd']:.1f}) & "
               f"{v2['w20']:.1f} & {v2['trm']:.1f} & {v2['n100']:.1f}    [med per seed V1 {' '.join(f'{x:.1f}' for x in v1['med_seeds'])}"
               f" | V2 {' '.join(f'{x:.1f}' for x in v2['med_seeds'])}]")
out += ['', 'columns as main-text Table 2: V1 Med. (s.d.) & +-20% & V2 Med. (s.d.) & +-20% & RMSPE99 & >100%']
open('results/s4_doseh/table2_doseevent_row.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
