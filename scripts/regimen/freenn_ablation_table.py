#!/usr/bin/env python3
"""Supplementary ablation of the FREE-NETWORK dose-event model on Baaz et al.'s regimen benchmark
12/10: Table 4 keeps only free flow + jump b(p) + noise 20 mg/L + dose-normalised encoder; the other variants go to the
supplement). Every cell is recomputed from the run JSONs with one definition:
  unseen   = mean over the 5 unseen regimens (amt350/600/850, time400/800) of the median |AUC(0-30 h) error| (%)
  observed = the same on the training regimen
  mean +- sd over seeds (ddof 1), n seeds; a run with unseen > 1000 % is DIVERGED, left out of the mean and reported.
No dose elasticity. Writes results/regimen_dh/freenn_ablation_table.txt and the LaTeX rows to
results/regimen_dh/freenn_ablation_rows.tex."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, json
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
R = os.path.join(ROOT, 'results/regimen_dh')
EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'
UNSEEN = ('amt350', 'amt600', 'amt850', 'time400', 'time800')
SEEDS = {'sparse': range(5), 'dense': range(3), 'titr': range(3)}

# (key, noise, jump, encoder, label of the structure) -- free drug network in the drug coordinates for every row but 'plain'
VARIANTS = [
    ('B',         'learned',          'population', 'raw'),
    ('Bfix',      '0.05 x max',       'population', 'raw'),
    ('Bfixkp',    '0.05 x max',       'b(p)',       'raw'),
    ('Bfix20',    '20 mg/L',          'population', 'raw'),
    ('Bfixkp20',  '20 mg/L',          'b(p)',       'raw'),
    ('Bfixkp20n', '20 mg/L',          'b(p)',       'dose-normalised'),
    ('plain',     'learned',          'full latent', 'raw'),
]
MAIN = 'Bfixkp20n'
PENDING = {('Bfixkp20n', 'lin', 'titr'), ('Bfixkp20n', 'lin', 'dense')}   # queue_t4norm.sh


def path(model, sim, col, s):
    mm, tag = sim == 'mm500', f'N100_{"sparse" if col in ("sparse", "titr") else col}_s{s}.json'
    if model == 'ebvae':
        if mm:
            return f'{EB}/res_mm500/{"titr0.8" if col == "titr" else "ctrl"}/{tag}'
        p = f'{EB}/res_titr/{"titr0.8" if col == "titr" else "ctrl"}/{tag}'
        return p if os.path.exists(p) or col == 'titr' else f'{EB}/res/N100_{col}' + (f'_s{s}' if s else '') + '.json'
    return f'{R}/{"mm500/" if mm else ""}{"titr0.8/" if col == "titr" else ""}{model}/{tag}'


def cell(model, sim, col):
    u, o, div = [], [], []
    for s in SEEDS[col]:
        p = path(model, sim, col, s)
        if not os.path.exists(p):
            continue
        ind = json.load(open(p))['individual']
        uu = float(np.mean([ind[g]['auc_abs_median'] for g in UNSEEN]))
        if not np.isfinite(uu) or uu > 1000:
            div.append(s); continue
        u.append(uu); o.append(float(ind['train']['auc_abs_median']))
    return dict(u=u, o=o, div=div)


def fmt(v, tex=False):
    if not v:
        return '---'
    sd = np.std(v, ddof=1) if len(v) > 1 else None
    pm = r'$\pm$' if tex else ' +- '
    s = f'{np.mean(v):.1f}' + (f'{pm}{sd:.1f}' if sd is not None else '')
    return s + (f'$^{{({len(v)})}}$' if tex else f' ({len(v)})')


COLS = [('lin', 'sparse', 'u'), ('lin', 'sparse', 'o'), ('lin', 'titr', 'u'), ('lin', 'dense', 'u'),
        ('mm500', 'sparse', 'u'), ('mm500', 'dense', 'u')]
HEAD = ['lin sparse unseen', 'lin sparse observed', 'lin titrated unseen', 'lin dense unseen', 'sat sparse unseen', 'sat dense unseen']

lines, tex = [__doc__.strip().split('\n')[0], ''], []
lines.append(f'{"model":44s} ' + ' | '.join(f'{h:>20s}' for h in HEAD))
rows = [('ebvae', 'EB-VAE (dose in the vector field), reference', None)] + [(k, f'{k:10s} noise {n:10s} jump {j:11s} enc {e}', (n, j, e)) for k, n, j, e in VARIANTS]
for key, lab, spec in rows:
    cs = {(sim, col): cell(key, sim, col) for sim, col, _ in COLS}
    txt = [fmt(cs[(sim, col)][w]) for sim, col, w in COLS]
    div = [f'{sim}/{col} s{d}' for (sim, col), c in cs.items() for d in c['div']]
    lines.append(f'{lab:44s} ' + ' | '.join(f'{t:>20s}' for t in txt) + (f'   DIVERGED: {", ".join(sorted(set(div)))}' if div else ''))
    t = [(r'\pend' if (key, sim, col) in PENDING and not cs[(sim, col)][w] else fmt(cs[(sim, col)][w], tex=True))
         for sim, col, w in COLS]
    if spec is None:
        tex.append(r'EB-VAE (reference) & & & & ' + ' & '.join(t) + r' \\')
    else:
        n, j, e = spec
        noise = {'learned': 'learned', '0.05 x max': r'$0.05\times$max', '20 mg/L': r'20~mg\,L$^{-1}$'}[n]
        jump = {'population': 'population', 'b(p)': r'$b(p)$', 'full latent': r'full state$^{a}$'}[j]
        enc = {'raw': 'raw', 'dose-normalised': 'dose-normalised'}[e]
        name = r'\textbf{main text}' if key == MAIN else ''
        tex.append(f'{name} & {noise} & {jump} & {enc} & ' + ' & '.join(t) + r' \\')
def seed_value(model, sim, col, s):
    p = path(model, sim, col, s)
    if not os.path.exists(p):
        return None
    ind = json.load(open(p))['individual']
    v = float(np.mean([ind[g]['auc_abs_median'] for g in UNSEEN]))
    return v if v <= 1000 else float('inf')
lines.append('')
lines.append('seed 1 (sparse) / seed 0 (dense) values of the multi-seed rows, to compare with the single-seed rows:')
for key in ('ebvae', 'B', 'Bfixkp20', 'Bfixkp20n'):
    lines.append(f'   {key:10s} lin sparse s1 {seed_value(key, "lin", "sparse", 1)}  sat sparse s1 {seed_value(key, "mm500", "sparse", 1)}  '
                 f'lin dense s0 {seed_value(key, "lin", "dense", 0)}')
open(f'{R}/freenn_ablation_table.txt', 'w').write('\n'.join(lines) + '\n')
open(f'{R}/freenn_ablation_rows.tex', 'w').write('\n'.join(tex) + '\n')
print('\n'.join(lines))
