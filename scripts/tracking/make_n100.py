#!/usr/bin/env python3
"""Training folder <cohort>_n100 for a regenerated cohort: the SAME 100 training patients as a reference
_n100 folder (rows taken from the new cohort's train CSV), the new cohort's full test CSV and truth, i.e. the layout of
confound_vc00_s4_pnoise_win_n100 (train 100 patients, test = the cohort's 1100 test patients).

    make_n100.py <new cohort> <reference n100 folder>     e.g. confound_vc00_s4_pnoise_win_shift70 confound_vc00_s4_pnoise_win_n100
"""
import os, sys, shutil
import pandas as pd

R = 'results/exp_film_run'
new, ref = sys.argv[1], sys.argv[2]
out = os.path.join(R, new + '_n100')
if os.path.exists(out):
    sys.exit(f'refusing: {out} exists')
ids = pd.read_csv(os.path.join(R, ref, 'virtual_cohort_film_train.csv')).ID.unique()
tr = pd.read_csv(os.path.join(R, new, 'virtual_cohort_film_train.csv'))
sub = tr[tr.ID.isin(ids)]
missing = set(ids) - set(sub.ID.unique())
if missing:
    sys.exit(f'{len(missing)} reference training IDs are not in {new}: {sorted(missing)[:10]}')
os.makedirs(out)
sub.to_csv(os.path.join(out, 'virtual_cohort_film_train.csv'), index=False)
shutil.copy(os.path.join(R, new, 'virtual_cohort_film_test.csv'), os.path.join(out, 'virtual_cohort_film_test.csv'))
shutil.copy(os.path.join(R, new, 'confound_truth.csv'), os.path.join(out, 'confound_truth.csv'))
print(f'{out}: train {sub.ID.nunique()} patients (IDs of {ref}), test copied from {new}')
