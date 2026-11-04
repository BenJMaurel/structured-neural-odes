#!/usr/bin/env python3
"""Tacrolimus regimen benchmark across sampling designs: sparse (6 samples, seeds 0-4), intensive PK visit (12, seeds 0-4), dense (61, seeds 0-2); scenario 4
(saturable) and scenario 2 (linear). Unseen = mean over the 5 unseen regimens of the median |AUC err| %; training = the observed
regimen; mean +- sd over seeds (ddof 1); paired = dose-event - EB-VAE on common seeds. Other baselines where they were run
(scenario 4 sparse / dense only; dose-cond UNTUNED, Lu iScience as published = encoder fed the regimen to predict).
--clip0 (20/10): every model's predictions clipped at zero, read from
results/regimen_dh/clip0/ (scripts/regimen/clip0_eval.py, which checks each run's unclipped reload against its JSON)."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, numpy as np
R = 'results/regimen_dh'; EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'
DESIGNS = (('sparse', '6', range(5)), ('visit', '12', range(5)), ('dense', '61', range(3)))
SEL = json.load(open(f'{R}/dc_tune/selected.json'))['selected'] if os.path.exists(f'{R}/dc_tune/selected.json') else None
TUNED = [] if SEL in (None, 'default') else [(f'dose-cond (tuned: {SEL})', f'dc_tuned_{SEL}')]
MODELS = {4: [('dose-event (ours)', f'{R}/tacro/Bfixkp4n'), ('EB-VAE', f'{EB}/res_tacro/ctrl')]
             + [(lab, f'{R}/tacro/{sub}') for lab, sub in TUNED] + [('dose-cond (untuned)', f'{R}/tacro/dosecond_n'),
              ('Lu iScience (as published)', f'{R}/tacro/luis'), ('Lu NMI', f'{R}/tacro/lunmi')],
          2: [('dose-event (ours)', f'{R}/tacro_s2/Bfixkp3.1n'), ('EB-VAE', f'{EB}/res_tacro_s2/ctrl')]
             + [(lab, f'{R}/tacro_s2/{sub}') for lab, sub in TUNED] + [('dose-cond (untuned)', f'{R}/tacro_s2/dosecond_n'),
             ('Lu iScience (as published)', f'{R}/tacro_s2/luis'), ('Lu NMI', f'{R}/tacro_s2/lunmi')]}   # Lu sc2 sparse: 16/10 (queue_tacro_s2_lu.sh)


CLIP = '--clip0' in sys.argv


def clip_tag(d):
    sc = 2 if ('tacro_s2' in d) else 4
    for key, tag in (('Bfixkp', 'de'), ('res_tacro', 'eb'), ('dc_tuned', 'dc'), ('luis', 'luis'), ('lunmi', 'lunmi')):
        if key in d:
            return f'{tag}_sc{sc}'
    return None


def load(d, design, seeds):
    out = {}
    if CLIP:
        tag = clip_tag(d)
        for s in seeds:
            p = f'{R}/clip0/{tag}_{design}_s{s}.json'
            if tag and os.path.exists(p):
                r = json.load(open(p))
                out[s] = (np.mean([v['clipped'] for k, v in r.items() if k != 'train']), r['train']['clipped'])
        return out
    for s in seeds:
        p = f'{d}/N100_{design}_s{s}.json'
        if os.path.exists(p):
            r = json.load(open(p))['individual']
            out[s] = (np.mean([v['auc_abs_median'] for k, v in r.items() if k != 'train']), r['train']['auc_abs_median'])
    return out


ms = lambda x: f'{np.mean(x):5.1f} +- {np.std(x, ddof=1) if len(x) > 1 else 0:4.1f}'
for sc, title in ((4, 'scenario 4 (saturable elimination)'), (2, 'scenario 2 (linear elimination)')):
    print(f'\n=== {title}')
    for design, n, seeds in DESIGNS:
        print(f'-- {design} ({n} samples per patient, seeds {list(seeds)})')
        res = {lab: load(d, design, seeds) for lab, d in MODELS[sc]}
        for lab, v in res.items():
            if v:
                print(f'   {lab:28s} n {len(v)}  unseen {ms([x[0] for x in v.values()])}  training {ms([x[1] for x in v.values()])}'
                      f'  per seed {[round(float(x[0]), 1) for x in v.values()]}')
        a, b = res['dose-event (ours)'], res['EB-VAE']; com = sorted(set(a) & set(b))
        if com:
            dd = [a[s][0] - b[s][0] for s in com]
            print(f'   paired dose-event - EB-VAE: {np.mean(dd):+.1f} pt, ours lower in {sum(x < 0 for x in dd)}/{len(dd)}')
