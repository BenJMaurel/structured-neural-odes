#!/usr/bin/env python3
"""Two-visit tacrolimus task (windowed scenario 4, N=100, ep6000, 1100 in-range test patients): per-seed median |AUC error| at V1
and V2 from the per_patient arrays of the eval JSONs, for the dose-event versions: rerun the
Fig. 3 model on 3 more seeds to check that it reproduces 21.7 / 24.6 before the ablation).
  dh7pg     population jump, encoder reads the dose         (Table 2 row 12.7 / 15.5; seeds 1-3)
  dh7pgkp   ABLATION: + jump b(p) only (encoder reads the dose)           (chain_doseh_ablation.sh, seeds 1-3)
  dh7pgen   ABLATION: + dose-normalised encoder only (population jump)    (chain_doseh_ablation.sh, seeds 1-3)
  dh7pgkpn  jump b(p) + dose-normalised encoder (Fig. 3)    (21.7 / 24.6 on seeds 1-3)
  dh7pgkpnd OPTION (b): Fig. 3 + --encoder-dose-value (dose value read through zero-initialised weights; )
Paired: per-seed difference to dh7pg on the same seed (same data, same initialisation stream); dh7pgkpnd also to dh7pgkp.
Writes results/s4_doseh/dosehkpn_summary.txt"""
import json, os
import numpy as np

E = 'results/s4_doseh/eval'
out = [__doc__.split('\n')[0], '']


def row(v, seeds):
    per = []
    for s in seeds:
        p = f'{E}/ep006000_{v}_osplit1-5_n100_s{s}_on_in.json'
        if not os.path.exists(p):
            continue
        q = json.load(open(p))['per_patient']
        e1 = np.array(q['pred_auc_v1']) / np.array(q['true_auc_v1']) - 1
        e2 = np.array(q['pred_auc_v2']) / np.array(q['true_auc_v2']) - 1
        per.append((s, 100 * np.median(np.abs(e1)), 100 * np.median(np.abs(e2)), 100 * np.mean(np.abs(e2) <= 0.2),
                    int(np.sum(np.abs(e2) > 1)), 100 * np.median(e2)))
    return per


base = {x[0]: x for x in row('dh7pg', (1, 2, 3))}
base_kp = {x[0]: x for x in row('dh7pgkp', (1, 2, 3))}
for v, seeds in (('dh7pg', (1, 2, 3)), ('dh7pgkp', (1, 2, 3)), ('dh7pgen', (1, 2, 3)), ('dh7pgkpn', (1, 2, 3)),
                 ('dh7pgkpnd', (1, 2, 3))):
    per = row(v, seeds)
    if not per:
        out.append(f'{v} seeds {seeds}: not evaluated yet'); continue
    a = np.array([x[1:] for x in per])
    sd = lambda c: f' +- {np.std(a[:, c], ddof=1):.1f}' if len(a) > 1 else ''
    out.append(f'{v:9s} seeds {[x[0] for x in per]}: V1 {a[:, 0].mean():.1f}{sd(0)}  V2 {a[:, 1].mean():.1f}{sd(1)}  '
               f'within 20 % V2 {a[:, 2].mean():.1f}  >100 % V2 {a[:, 3].mean():.1f}/1100  bias V2 {a[:, 4].mean():+.1f}   per seed V1/V2 '
               + ' '.join(f's{x[0]} {x[1]:.1f}/{x[2]:.1f}' for x in per))
    if v != 'dh7pg':
        d = [(x[1] - base[x[0]][1], x[2] - base[x[0]][2]) for x in per if x[0] in base]
        if d:
            out.append(f'          paired minus dh7pg (same seed): V1 ' + ' '.join(f'{a:+.1f}' for a, _ in d)
                       + '   V2 ' + ' '.join(f'{b:+.1f}' for _, b in d))
    if v == 'dh7pgkpnd':
        d = [(x[1] - base_kp[x[0]][1], x[2] - base_kp[x[0]][2]) for x in per if x[0] in base_kp]
        if d:
            out.append(f'          paired minus dh7pgkp (same seed): V1 ' + ' '.join(f'{a:+.1f}' for a, _ in d)
                       + '   V2 ' + ' '.join(f'{b:+.1f}' for _, b in d)
                       + f'   (mean V1 {np.mean([a for a, _ in d]):+.1f}, V2 {np.mean([b for _, b in d]):+.1f})')
out.append('')
out.append('median |AUC err| % (pred/true - 1), in-range test cohort; mean +- sd over seeds.')
open('results/s4_doseh/dosehkpn_summary.txt', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
