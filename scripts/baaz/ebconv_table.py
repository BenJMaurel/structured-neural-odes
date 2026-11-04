#!/usr/bin/env python3
"""EB-VAE convergence + scenario 3. Works on partial results.
A. Part 2 dose-switch cohort, test_in (1261 patients), seeds 1-3: V1 / V2 metrics by stage-2 epoch (300 = their default and
   the published Part 2 run: asserted IDENTICAL per patient to results/baaz_part2/baaz/eval_test_in_seed<s>.json), the
   training MSE at the end of each chunk (from the log), and OT-FiLM (Part 2, ep6000) as reference.
B. NMI 3-scenario study, scenario 3, seeds 1-10 (40 test patients each): EB-VAE by stage-2 epoch vs the plain latent ODE (final
   ckpt = primary; _best = test-selected, as in Table 1) and MAP-BE (SIGMA fixed, published init) of the SAME runs
   (results/scen_rerun/runs/s3_seedNNN). True AUCs asserted equal across methods. Per run RMSPE / MPE / median |err|,
   mean +- s.d. over runs and paired tests over runs (Table 1's format); pooled median |err|, RMSPE w/o worst 1 %, n > 100 %.
    ebconv_table.py  -> results/baaz_scen3/ebconv_table.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, re, sys, json, glob, numpy as np, pandas as pd
from scipy import stats
os.chdir(REPO); sys.argv = sys.argv[:1]; sys.path.insert(0, 'scripts/ndm')
from ndm_tables import metrics
out = []; P = lambda s='': out.append(s)
fmt = lambda m: f'med {m["med"]:5.1f}  bias {m["bias"]:+6.1f}  w20 {m["w20"]:5.1f}  trm {m["trm"]:6.1f}  n>100% {m["n100"]:4d}'

P('A. EB-VAE on the Part 2 dose-switch cohort (test_in, 1261 patients): was training still improving after their 300 epochs?')
for s in (1, 2, 3):
    files = sorted(glob.glob(f'results/baaz_part2/curve/eval_test_in_seed{s}_ep*.json'), key=lambda f: int(re.search(r'_ep(\d+)', f).group(1)))
    if not files:
        P(f'  seed {s}: no checkpoint yet'); continue
    log = open(f'logs/baaz/ebcurve_s{s}.log').read() if os.path.exists(f'logs/baaz/ebcurve_s{s}.log') else ''
    st2 = log[log.find('stage 1 done'):]
    for f in files:
        E = int(re.search(r'_ep(\d+)', f).group(1)); pp = json.load(open(f))['per_patient']
        chunk = st2[:st2.find(f'stage 2 epoch {E} done')]; last = re.findall(r'Epoch (\d+), MSE ([0-9.]+).*?Add\. error: ([0-9.]+).*?lr: ([0-9.e-]+)', chunk)
        tr = f'train MSE {float(last[-1][1]):7.1f}  noise {last[-1][2]}  lr {last[-1][3]}' if last else ''
        tag = ''
        if E == 300:
            ref = json.load(open(f'results/baaz_part2/baaz/eval_test_in_seed{s}.json'))['per_patient']
            d = max(np.abs(np.array(ref[k]) - np.array(pp[k])).max() for k in ('pred_auc_v1', 'pred_auc_v2'))
            assert d < 1e-6, f'seed {s}: epoch 300 does not reproduce the published Part 2 run (max diff {d})'
            tag = '  (= published Part 2 run)'
        P(f'  seed {s} ep {E:4d}  V1 {fmt(metrics(pp["true_auc_v1"], pp["pred_auc_v1"]))} | V2 {fmt(metrics(pp["true_auc_v2"], pp["pred_auc_v2"]))}   {tr}{tag}')
for s in (1, 2, 3):
    f = f'results/baaz_part2/eval/ep006000_film_n100_s{s}_on_in.json'
    if os.path.exists(f):
        pp = json.load(open(f))['per_patient']
        P(f'  OT-FiLM s{s} ep6000  V1 {fmt(metrics(pp["true_auc_v1"], pp["pred_auc_v1"]))} | V2 {fmt(metrics(pp["true_auc_v2"], pp["pred_auc_v2"]))}')

P(); P('B. Scenario 3 (MM elimination, HT 35; factual AUC of the observed interval from 0/1/3 h samples; 160 train / 40 test per run)')
rows = {}
for s in range(1, 11):
    R = f'results/scen_rerun/runs/s3_seed{s:03d}'
    lode = {}
    for c in ('final', 'best'):
        j = json.load(open(f'{R}/lode_{c}.json'))['per_series']
        lode[c] = pd.DataFrame(dict(true=j['true_auc'], pred=j['pred_auc']), index=[int(i) for i in j['id']])
    x = pd.read_csv(f'{R}/tacro_mapbayest_auc_sigvar.csv').set_index('ID')
    meths = {'Latent ODE, final ckpt': lode['final'], 'Latent ODE, _best (test-selected)': lode['best'],
             'MAP-BE (SIGMA fixed)': pd.DataFrame(dict(true=x.AUC_observed, pred=x.auc_ipred))}
    for f in glob.glob(f'results/baaz_scen3/runs/eval_test_s{s}_seed{s}_ep*.json'):
        E = int(re.search(r'_ep(\d+)', f).group(1)); pp = json.load(open(f))['per_patient']
        meths[f'EB-VAE, {E} ep'] = pd.DataFrame(dict(true=pp['true_auc'], pred=pp['pred_auc']), index=pp['id'])
    ids = sorted(set.intersection(*[set(m.index) for m in meths.values()]))
    assert len(ids) == 40, (s, len(ids))
    for k, m in meths.items():
        assert np.allclose(m.loc[ids, 'true'].values, lode['final'].loc[ids, 'true'].values, rtol=1e-4), (s, k, 'true AUC differs')
        rows.setdefault(k, {})[s] = (m.loc[ids, 'pred'] / m.loc[ids, 'true'] - 1).values
eb_keys = sorted([k for k in rows if k.startswith('EB-VAE')], key=lambda k: int(k.split(', ')[1].split()[0]))
order = ['Latent ODE, final ckpt', 'Latent ODE, _best (test-selected)', 'MAP-BE (SIGMA fixed)'] + eb_keys
P(f'  {"method":36s} runs  RMSPE mean+-sd    MPE mean+-sd    | pooled: med |err|  RMSPE-1%  n>100%')
for k in order:
    r = rows[k]; rm = [100 * np.sqrt(np.mean(v ** 2)) for v in r.values()]; mp = [100 * np.mean(v) for v in r.values()]
    pool = np.concatenate(list(r.values())); a = np.abs(pool); keep = a <= np.quantile(a, 0.99)
    P(f'  {k:36s} {len(r):4d}  {np.mean(rm):6.2f} +- {np.std(rm, ddof=1) if len(rm) > 1 else 0:5.2f}  {np.mean(mp):+6.2f} +- {np.std(mp, ddof=1) if len(mp) > 1 else 0:5.2f}'
      f'  |  {100 * np.median(a):6.1f}  {100 * np.sqrt(np.mean(pool[keep] ** 2)):8.1f}  {int((a > 1).sum()):5d}')
P('  paired over runs (EB-VAE minus comparator; per-run RMSPE and per-run median |err|; t-test / Wilcoxon):')
for k in eb_keys:
    for c in ('Latent ODE, final ckpt', 'Latent ODE, _best (test-selected)', 'MAP-BE (SIGMA fixed)'):
        S = sorted(set(rows[k]) & set(rows[c]))
        if len(S) < 3: continue
        for lab, fn in (('RMSPE', lambda v: 100 * np.sqrt(np.mean(v ** 2))), ('med|err|', lambda v: 100 * np.median(np.abs(v)))):
            d = np.array([fn(rows[k][s]) - fn(rows[c][s]) for s in S])
            P(f'    {k:16s} vs {c:34s} {lab:8s} {np.mean(d):+6.2f} pt  (EB-VAE better in {int((d < 0).sum())}/{len(S)})  '
              f't p = {stats.ttest_1samp(d, 0).pvalue:.2g}  Wilcoxon p = {stats.wilcoxon(d).pvalue:.2g}')
txt = '\n'.join(out); print(txt); os.makedirs('results/baaz_scen3', exist_ok=True); open('results/baaz_scen3/ebconv_table.txt', 'w').write(txt + '\n')
