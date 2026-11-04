#!/usr/bin/env python3
"""-2LL (OFV, importance sampling) of the published-init / default-init / auto-init Monolix fits on the same dataset
(chain_loglik.sh). dOFV = OFV(fit) - OFV(published-init fit): negative = that fit explains the training data BETTER.
Plausible fit: Vc_pop 100-2000 L, KTR_pop 1-10 /h, Vp_pop 50-5000 L, Q_pop 5-1000 L/h. Importance-sampling OFV carries
Monte Carlo noise of order 1 point; |dOFV| < 2 is treated as a tie."""
import glob, json, os, collections
import numpy as np
L = 'results/scen_rerun/loglik'
R = collections.defaultdict(dict)
for f in glob.glob(f'{L}/*_*.json'):
    i, fit = os.path.basename(f)[:-5].split('_', 1); j = json.load(open(f)); R[int(i)][fit] = j
def plaus(p): return 100 <= p['Vc_pop'] <= 2000 and 1 <= p['KTR_pop'] <= 10 and 50 <= p['Vp_pop'] <= 5000 and 5 <= p['Q_pop'] <= 1000
for sc in (1, 2, 3):
    ids = sorted(i for i in R if i // 10000 == sc)
    if not ids: continue
    print('=' * 100); print(f'SCENARIO {sc}: {len(ids)} datasets')
    print(f'  {"dataset":8s} {"OFV published":>14s} | {"dOFV default":>12s} {"fit":>10s} | {"dOFV auto":>10s} {"fit":>10s}')
    tally = {f: collections.Counter() for f in ('definit', 'autoinit')}
    for i in ids:
        r = R[i]; base = r.get('published'); line = f'  {i:8d} {base["ofv"] if base else float("nan"):14.1f} |'
        for f in ('definit', 'autoinit'):
            if f in r and base:
                d = r[f]['ofv'] - base['ofv']; pl = plaus(r[f]['params'])
                line += f' {d:+12.1f} {"plausible" if pl else "DEGENERATE":>10s} |'
                tally[f][(('plaus' if pl else 'degen'), 'better' if d < -2 else ('worse' if d > 2 else 'tie'))] += 1
            else: line += f' {"-":>12s} {"":>10s} |'
        print(line)
    for f, lab in (('definit', 'default init'), ('autoinit', 'auto-init')):
        t = tally[f]
        for k in ('degen', 'plaus'):
            n = sum(v for (a, _), v in t.items() if a == k)
            if n: print(f'  {lab:12s} {("degenerate" if k == "degen" else "plausible"):10s} fits: {n:2d} | better likelihood than published-init fit in '
                        f'{t[k, "better"]}, tie {t[k, "tie"]}, worse {t[k, "worse"]}')
