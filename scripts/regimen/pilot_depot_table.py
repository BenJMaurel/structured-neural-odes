#!/usr/bin/env python3
"""Pilot table, "Fix the model and rerun"): for each pilot run of results/regimen_dh/pilot_depot/ and the existing run of
the same design and seed, accuracy and curve shape, to choose the fix before any full rerun.
  unseen    mean over the original unseen regimens of the median |AUC err| % (Baaz design: also the five held-out ones)
  training  median |AUC err| % on the training regimen
  below 0   share of test patients whose predicted curve goes below zero (training regimen and the highest unseen dose)
  AUC/mg    dose proportionality, see the note under the table
  dip       median over patients of the lowest prediction in [last dose - 0.25 h, last dose + 0.5 h] minus the trough just before
  jump      median share of the rise to the peak that happens within 15 min of the last dose (truth ~0.06 for tacrolimus)
  t_peak    median time from the last dose to the predicted maximum (truth 1.25 h for tacrolimus)
Models rebuilt with dh_regimen.py run as a module (the run's own flags), weights loaded, reload CHECKED against each run's JSON.
Writes results/regimen_dh/pilot_depot/pilot_table.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import os, sys, json, runpy, io, contextlib
os.environ['TACRO_SCENARIO'] = '2'; os.environ['REGIMENS_CONFIRM'] = '1'
import numpy as np, torch
ROOT = REPO; os.chdir(ROOT)
SCR = os.environ.get('PILOT_SCRATCH', 'results/scratch/pilot_scratch'); os.makedirs(SCR, exist_ok=True)
P = 'results/regimen_dh/pilot_depot'
RUNS = [('tacro 12 samples', 'visit', 0, 'results/regimen_dh/tacro_s2/Bfixkp3.1n/N100_visit_s0', 'tacro_s2_visit_{v}_s0'),
        ('tacro 6 samples', 'sparse', 0, 'results/regimen_dh/tacro_s2/Bfixkp3.1n/N100_sparse_s0', 'tacro_s2_sparse_{v}_s0'),
        ('Baaz sparse s5', 'sparse', 5, 'results/regimen_dh/confirm/ctrl/Bfixkp20n/N100_sparse_s5', 'baaz_ctrl_sparse_{v}_s5'),
        ('Baaz sparse s6', 'sparse', 6, 'results/regimen_dh/confirm/ctrl/Bfixkp20n/N100_sparse_s6', 'baaz_ctrl_sparse_{v}_s6')]
MULT = (0.25, 0.5, 1.5, 2, 3)   # dose-proportionality check: single dose from each patient's drug-free state, multiples of the low arm
torch.set_num_threads(1)


def rebuild(pt):
    ck = torch.load(pt, map_location='cpu', weights_only=False); c = ck['cli']
    argv = ['x', str(c['n_per_arm']), c['design'], str(c['seed']), f'{SCR}/u.json', '--no-check', '--iters', '1', '--enc-dose', c['enc_dose']]
    argv += ['--obsrv-mgl', str(c['obsrv_mgl'])] if c.get('obsrv_mgl') is not None else []
    argv += ['--kick-phys', str(c['kick_phys'])] if c.get('kick_phys') else []
    argv += ['--sim', 'tacro'] if c.get('sim') == 'tacro' else []
    argv += ['--readout-relu'] if c.get('readout_relu') else []
    argv += ['--depot', str(c['depot'])] if c.get('depot') else []
    argv += ['--depot-linear'] if c.get('depot_linear') else []
    sys.argv = argv
    with contextlib.redirect_stdout(io.StringIO()):
        g = runpy.run_path('scripts/regimen/dh_regimen.py', run_name='pilot')
    g['model'].load_state_dict(ck['state_dict']); g['model'].eval()
    return g, c


def stats(base):
    g, c = rebuild(base + '.pt'); res = json.load(open(base + '.json'))['individual']
    EG = g['EG']; reg = g['regs']
    with torch.no_grad():
        mu, _ = g['model'].encoder_z0(g['X_te'], g['tp_samp'], static=g['S_te'], run_backwards=True)
        pred = {r: g['decode'](g['simulate'](mu, *reg[r], EG))[0].numpy() * g['YMAX'] for r in reg}
    truth = {r: np.stack([np.interp(EG, g['t_eval'], cc) for cc in g['truth'][r]]) for r in reg}
    for r in reg:   # reload check against the run's own per-regimen median
        e = 100 * np.median(np.abs(np.trapezoid(pred[r], EG, axis=1) / np.trapezoid(truth[r], EG, axis=1) - 1))
        assert abs(e - res[r]['auc_abs_median']) < 1e-2, f'{base} {r}: reload {e:.3f} vs run {res[r]["auc_abs_median"]:.3f}'
    orig = [r for r in reg if r != 'train' and not r.startswith('c_')]; held = [r for r in reg if r.startswith('c_')]
    hi = max((r for r in orig if r.startswith('amt')), key=lambda r: float(r[3:]))
    t_last = float(max(reg['train'][0])); i0 = int(np.argmin(np.abs(EG - (t_last - 0.25))))
    w = (EG >= t_last - 1e-9) & (EG <= t_last + 6 + 1e-9); near = (EG >= t_last - 0.25 - 1e-9) & (EG <= t_last + 0.5 + 1e-9)
    i15 = int(np.argmin(np.abs(EG - (t_last + 0.25))))
    def shape(c):
        seg = c[:, w]; rise = seg.max(1) - c[:, i0]
        return dict(dip=float(np.median(c[:, near].min(1) - c[:, i0])), jump=float(np.nanmedian((c[:, i15] - c[:, i0]) / np.where(rise > 1e-6, rise, np.nan))),
                    tpeak=float(np.median(EG[w][np.argmax(seg, 1)] - t_last)))
    with torch.no_grad():   # drug-attributable AUC per mg relative to the low training dose (truth 1.00 for a linear model)
        zf = mu @ g['model'].orth_split.P_phys(); n = zf.shape[1]
        auc = lambda d: np.trapezoid(g['decode'](g['simulate'](zf, [0], np.full((n, 1), d), EG))[0].numpy() * g['YMAX'], EG, axis=1)
        a0, a1 = auc(0.0), auc(float(g['LO']))
        prop = {m: float(np.median((auc(m * g['LO']) - a0) / m / (a1 - a0))) for m in MULT}
    return dict(prop=prop, unseen=np.mean([res[r]['auc_abs_median'] for r in orig]), held=np.mean([res[r]['auc_abs_median'] for r in held]) if held else np.nan,
                train=res['train']['auc_abs_median'], below=float(np.mean((pred['train'].min(1) < 0) | (pred[hi].min(1) < 0))),
                **shape(pred['train']), truth=shape(truth['train']))


L = [__doc__.split('\n')[0], '']
for title, design, seed, orig, pat in RUNS:
    L.append(f'=== {title}')
    rows = [('current model', orig)] + [(f'+ {lab}', f'{P}/{pat.format(v=v)}') for v, lab in (('relu', 'non-negative readout'), ('depot', 'depot (1 coordinate)'), ('deplin', 'linear depot'))]
    tr = None
    for lab, base in rows:
        if not os.path.exists(base + '.json'):
            L.append(f'   {lab:26s} (not available)'); continue
        s = stats(base); tr = s['truth']
        L.append(f'   {lab:26s} unseen {s["unseen"]:5.1f}' + (f'  held-out {s["held"]:5.1f}' if not np.isnan(s['held']) else '') +
                 f'  training {s["train"]:5.1f}  below 0 {100 * s["below"]:3.0f}%  dip {s["dip"]:+6.2f}  jump {s["jump"]:5.2f}  t_peak {s["tpeak"]:4.2f} h'
                 + '  | AUC/mg vs low arm ' + ' '.join(f'x{m:g}:{v:.2f}' for m, v in s['prop'].items()))
    if tr:
        L.append(f'   {"truth":26s} dip {tr["dip"]:+6.2f}  jump {tr["jump"]:5.2f}  t_peak {tr["tpeak"]:4.2f} h')
    L.append('')
L.append('AUC/mg vs low arm: single dose of m x the low training dose (Baaz 400 mg, tacrolimus 2 mg) from each patient\'s drug-free state, '
         'drug-attributable AUC per mg over that of the low dose, median over test patients; 1.00 at every m for these linear simulators')
open(f'{P}/pilot_table.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
