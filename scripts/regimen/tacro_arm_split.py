#!/usr/bin/env python3
"""Tacrolimus regimen benchmark: median SIGNED AUC error per regimen, split by the arm the test patient was OBSERVED on (even index
2 mg, odd 4 mg, in both runners), dose-event vs EB-VAE, scenarios 2 / 4 x dense / visit / sparse. An arm split
(over-prediction of 4-mg patients at lower amounts, under-prediction of 2-mg patients at higher ones) = the dose effect attributed to
the patients (Prop. priorgauge of docs/dose_injection_theory.tex)."""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import json, os, numpy as np
R = 'results/regimen_dh'; EB = f'{REPO}/third_party/baaz_ebvae/ours/regimen'
REGS = ('train', 'amt1.75', 'amt3', 'amt4.25', 'timeLOW', 'timeHIGH')
for sc, ours, eb in ((2, f'{R}/tacro_s2/Bfixkp3.1n', f'{EB}/res_tacro_s2/ctrl'), (4, f'{R}/tacro/Bfixkp4n', f'{EB}/res_tacro/ctrl')):
    for d, seeds in (('dense', range(3)), ('visit', range(5)), ('sparse', range(5))):
        print(f'== scenario {sc} {d}: median SIGNED AUC error %, mean over seeds, split by the arm the test patient was OBSERVED on (2 mg | 4 mg)')
        for lab, base in (('dose-event', ours), ('EB-VAE', eb)):
            rows = {g: [] for g in REGS}
            for s in seeds:
                p = f'{base}/N100_{d}_s{s}.json'
                if not os.path.exists(p): continue
                r = json.load(open(p))['individual']
                for g in REGS:
                    e = 100 * np.array(r[g]['auc_err_per_patient'])
                    rows[g].append((np.median(e[0::2]), np.median(e[1::2])))
            print(f'   {lab:10s} ' + '  '.join(f'{g} {np.mean([x[0] for x in v]):+5.1f}|{np.mean([x[1] for x in v]):+5.1f}' for g, v in rows.items()))
