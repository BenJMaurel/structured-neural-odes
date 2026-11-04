#!/usr/bin/env python3
"""Scenario 3, 10 cohorts (160 train / 40 test): (A) is the EB-VAE's train/test gap an amortisation gap? map_refine.py re-fits
each patient's latent by MAP on the same 0/1/3 h samples the encoder sees, model frozen, with the model's own noise or the true
noise; (B) does OUR plain latent ODE have a comparable train/test gap? eval_lode.py EVAL_SPLIT=train vs test, final and _best
(test-selected) checkpoints. Per-cohort median |AUC err| and RMSPE (factual AUC of the observed interval), mean +- sd over
cohorts; paired Wilcoxon over cohorts. EB-VAE = published configuration at its default 300 stage-2 epochs.

    gap_table.py > results/baaz_check/gap_table.txt
"""
import os, json
import numpy as np
from scipy import stats

M = 'results/baaz_check/mapref'; L = 'results/baaz_check/lode_split'


def st(pp):
    e = np.array(pp['pred_auc']) / np.array(pp['true_auc']) - 1
    return 100 * np.median(np.abs(e)), 100 * np.sqrt(np.mean(e ** 2))


def ms(v):
    return f'{np.mean(v):5.1f} +- {np.std(v, ddof=1):4.1f}'


def paired(a, b):
    d = np.array(a) - np.array(b)
    return f'{np.mean(d):+5.1f} pt (lower in {int((d < 0).sum())}/{len(d)}, Wilcoxon p {stats.wilcoxon(d).pvalue:.2g})'


def main():
    print(__doc__.strip().split('\n\n')[0])
    E = {}
    for s in range(1, 11):
        for w in ('train', 'test'):
            f = f'{M}/scen3_s{s}_ep300_on_{w}.json'
            if os.path.exists(f):
                j = json.load(open(f))
                for tag, pp in j['per_patient'].items():
                    E.setdefault((tag, w), {})[s] = st(pp)
                for tag in ('map_model_noise', 'map_true_noise'):
                    E.setdefault((tag + '_diag', w), {})[s] = (j['summary'][tag]['obj_increased'], j['summary'][tag]['last_step_max'])
    print('\n== A. EB-VAE (published, ep300): encoder vs MAP-refined latent, training vs test patients')
    print(f'   {"latent":34s} {"train median":>15s} {"test median":>15s} {"train RMSPE":>15s} {"test RMSPE":>15s}   cohorts')
    names = {'encoder': 'encoder (amortised, as published)', 'map_model_noise': 'MAP, model\'s own learned noise', 'map_true_noise': 'MAP, true noise 0.71 + 0.113 C'}
    for tag, nm in names.items():
        tr, te = E.get((tag, 'train'), {}), E.get((tag, 'test'), {})
        S = sorted(set(tr) & set(te))
        if not S:
            continue
        print(f'   {nm:34s} {ms([tr[s][0] for s in S]):>15s} {ms([te[s][0] for s in S]):>15s} {ms([tr[s][1] for s in S]):>15s} {ms([te[s][1] for s in S]):>15s}   {len(S)}')
    for tag in ('map_model_noise', 'map_true_noise'):
        for w in ('train', 'test'):
            a, b = E.get((tag, w), {}), E.get(('encoder', w), {})
            S = sorted(set(a) & set(b))
            if len(S) >= 5:
                print(f'   {names[tag]:34s} minus encoder, {w:5s}: median {paired([a[s][0] for s in S], [b[s][0] for s in S])}; '
                      f'RMSPE {paired([a[s][1] for s in S], [b[s][1] for s in S])}')
        d = E.get((tag + '_diag', 'test'), {})
        if d:
            print(f'   {names[tag]:34s} convergence: objective increased for {sum(v[0] for v in d.values())} patients; max last step {max(v[1] for v in d.values()):.1e}')
    # ---------------------------------------------------------------- B
    print('\n== B. Our plain latent ODE (paper config): training vs test series, same prediction rule (encoder on 0/1/3 h)')
    G = {}
    for c in ('final', 'best'):
        tr, te = {}, {}
        for s in range(1, 11):
            ft, fe = f'{L}/s3_seed{s:03d}_{c}_train.json', f'{L}/s3_seed{s:03d}_{c}_test.json'
            if os.path.exists(ft) and os.path.exists(fe):
                a, b = json.load(open(ft)), json.load(open(fe))
                tr[s] = (a['v1']['median_abs'], a['v1']['rmspe'], a['n_series']); te[s] = (b['v1']['median_abs'], b['v1']['rmspe'], b['n_series'])
        S = sorted(set(tr) & set(te))
        if not S:
            continue
        G[c] = {s: te[s][0] - tr[s][0] for s in S}
        print(f'   latent ODE, {c:5s} ckpt  train median {ms([tr[s][0] for s in S])}  test median {ms([te[s][0] for s in S])}  '
              f'train RMSPE {ms([tr[s][1] for s in S])}  test RMSPE {ms([te[s][1] for s in S])}  gap (median) {ms([G[c][s] for s in S])}  '
              f'[n train / test series {tr[S[0]][2]} / {te[S[0]][2]}; {len(S)} cohorts]')
    tr, te = E.get(('encoder', 'train'), {}), E.get(('encoder', 'test'), {})
    for c in G:
        S = sorted(set(G[c]) & set(tr) & set(te))
        if len(S) >= 5:
            ge = [te[s][0] - tr[s][0] for s in S]
            print(f'   train/test gap (median |err|), EB-VAE encoder minus latent ODE {c}: {paired(ge, [G[c][s] for s in S])}')


if __name__ == '__main__':
    main()
