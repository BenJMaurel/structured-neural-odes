#!/usr/bin/env python3
"""Job 1 table (review 22/10): tacrolimus regimen benchmark with the 12 samples around the FIRST dose ('visit1') and the positive
(softplus) readout of the dose-event model, next to the previous 12-sample design ('visit', profile after the 4th dose).
Cells read from results/regimen_dh/clip0/ (scripts/regimen/clip0_eval.py: every model rebuilt or its curves read, unclipped value checked
against the run's JSON, then predictions clipped at zero; the softplus readout is positive by construction). Unseen = mean over the five
unseen regimens of the median |AUC(0-45 h) err| %, training = the observed regimen, mean +- sd over seeds; 'lower' = seeds in which the
dose-event model's unseen error is below the EB-VAE's. Lu regimen model under both encoder protocols (target = as in the draft; factual =
as Lu et al.'s code predicts a counterfactual regimen). Writes results/review22/visit1_table.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import glob, json, os, re, numpy as np
os.chdir(REPO)
C = 'results/regimen_dh/clip0'
ROWS = [('Dose events (ours)', 'de', ''), ('EB-VAE', 'eb', ''), ('Dose input (ablation)', 'dc', ''),
        ('Lu regimen, target encoder', 'luis', ''), ('Lu regimen, factual encoder', 'luis', 'fact'), ('Lu two-port', 'lunmi', '')]


def cell(model, sc, design, sfx):
    out = {}
    for f in glob.glob(f'{C}/{model}_sc{sc}_{design}_s*{sfx}.json'):
        m = re.search(rf'{model}_sc{sc}_{design}_s(\d+){re.escape(sfx)}\.json$', os.path.basename(f))
        if not m: continue
        r = json.load(open(f)); un = [k for k in r if k != 'train']
        out[int(m.group(1))] = (np.mean([r[k]['clipped'] for k in un]), r['train']['clipped'])
    return out


ms = lambda x: f'{np.mean(x):5.1f} +- {np.std(x, ddof=1) if len(x) > 1 else 0:4.1f}'
L = [__doc__.split('\n')[0], '']
for design, label, base in (('visit1', '12 samples around the first dose (new)', '_v1'), ('visit', '12 samples after the 4th dose (draft)', '')):
    for sc, el in ((2, 'linear'), (4, 'saturable')):
        L.append(f'-- {label}, {el} elimination')
        res = {}
        for lab, model, extra in ROWS:
            sfx = base + extra if base else ('_' + extra if extra else '')   # visit1: _v1 / _v1fact; visit: '' / _fact
            v = cell(model, sc, design, sfx); res[lab] = v
            if v:
                L.append(f'   {lab:30s} n {len(v)}  unseen {ms([x[0] for x in v.values()])}  training {ms([x[1] for x in v.values()])}')
        a, b = res['Dose events (ours)'], res['EB-VAE']; com = sorted(set(a) & set(b))
        if com:
            L.append(f'   lower (ours vs EB-VAE): {sum(a[s][0] < b[s][0] for s in com)}/{len(com)}')
        L.append('')
open('results/review22/visit1_table.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
