#!/usr/bin/env python3
"""Dose-cond after tuning (phase 2, queue_dc_phase2.sh; ) against the untuned dose-cond, the dose-event model and the
EB-VAE, on every regimen benchmark: Baaz et al.'s linear simulator (sparse s0-4, dense s0-2, titrated sparse s0-2) and the
tacrolimus regimen benchmark, scenario 4 (saturable) and scenario 2 (linear), sparse s0-4 and dense s0-2.
Per model: unseen-regimen error (mean over the 5 unseen regimens of the median |AUC err| %) and training-regimen error, mean +- sd
over seeds (ddof 1), per-seed unseen; paired difference tuned dose-cond - dose-event on common seeds."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, json, numpy as np
R = 'results/regimen_dh'; EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'
C = json.load(open(f'{R}/dc_tune/selected.json'))['selected'] if os.path.exists(f'{R}/dc_tune/selected.json') else None


def eb_baaz(d, col):
    def f(s):
        tag = f'N100_{d}_s{s}.json'; p = f'{EB}/res_titr/{"titr0.8" if col == "titr" else "ctrl"}/{tag}'
        return p if os.path.exists(p) or col == 'titr' else f'{EB}/res/N100_{d}' + (f'_s{s}' if s else '') + '.json'
    return f


BENCH = []   # (title, design, seeds, {label: path(s)})
for col, d, seeds in (('sparse', 'sparse', range(5)), ('dense', 'dense', range(3)), ('titr', 'sparse', range(3))):
    t = 'titr0.8/' if col == 'titr' else ''
    BENCH.append((f"Baaz linear simulator, {'titrated sparse' if col == 'titr' else d}", seeds, {
        'dose-cond TUNED': (lambda s, d=d, t=t: (f'{R}/dc_tune/{C}/N100_{d}_s{s}.json' if (d == 'sparse' and not t) else f'{R}/dc_tuned/{C}/{t}N100_{d}_s{s}.json')),
        'dose-cond (untuned)': (lambda s, d=d, t=t: f'{R}/dosecond_n/{t}N100_{d}_s{s}.json'),
        'dose-event (Bfixkp20n)': (lambda s, d=d, t=t: f'{R}/{t}Bfixkp20n/N100_{d}_s{s}.json'),
        'EB-VAE': eb_baaz(d, col)}))
for sc, sub, de, ebd in ((4, 'tacro', 'Bfixkp4n', 'res_tacro'), (2, 'tacro_s2', 'Bfixkp3.1n', 'res_tacro_s2')):
    for d, seeds in (('sparse', range(5)), ('dense', range(3))):
        BENCH.append((f'tacrolimus regimen, scenario {sc} ({"saturable" if sc == 4 else "linear"}), {d}', seeds, {
            'dose-cond TUNED': (lambda s, d=d, sub=sub: f'{R}/{sub}/dc_tuned_{C}/N100_{d}_s{s}.json'),
            'dose-cond (untuned)': (lambda s, d=d, sub=sub: f'{R}/{sub}/dosecond_n/N100_{d}_s{s}.json'),
            'dose-event': (lambda s, d=d, sub=sub, de=de: f'{R}/{sub}/{de}/N100_{d}_s{s}.json'),
            'EB-VAE': (lambda s, d=d, ebd=ebd: f'{EB}/{ebd}/ctrl/N100_{d}_s{s}.json')}))


def scores(path, seeds):
    out = {}
    for s in seeds:
        p = path(s)
        if os.path.exists(p):
            r = json.load(open(p))['individual']
            out[s] = (np.mean([v['auc_abs_median'] for k, v in r.items() if k != 'train']), r['train']['auc_abs_median'])
    return out


ms = lambda x: f'{np.mean(x):5.1f} +- {np.std(x, ddof=1) if len(x) > 1 else 0:4.1f}'
print(f'Selected dose-cond config: {C}')
for title, seeds, models in BENCH:
    print(f'\n=== {title} (seeds {list(seeds)})')
    sc = {lab: scores(f, seeds) for lab, f in models.items()}
    for lab, v in sc.items():
        if not v:
            print(f'   {lab:24s} n/a'); continue
        u = [x[0] for x in v.values()]; t = [x[1] for x in v.values()]
        print(f'   {lab:24s} n {len(v)}  unseen {ms(u)}  training regimen {ms(t)}  per seed {[round(float(x), 1) for x in u]}')
    a, b = sc.get('dose-cond TUNED', {}), next((sc[k] for k in sc if k.startswith('dose-event')), {})
    com = sorted(set(a) & set(b))
    if com:
        dd = [a[s][0] - b[s][0] for s in com]
        print(f'   paired tuned dose-cond - dose-event: {np.mean(dd):+.1f} pt, dose-event lower in {sum(x > 0 for x in dd)}/{len(dd)}')
