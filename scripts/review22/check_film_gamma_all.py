#!/usr/bin/env python3
"""Sign of the OT-FiLM scale gamma(d1, d2) in every final (ep6000) OT-FiLM checkpoint under the results roots that back the paper
(review 22/10; Table 2 alone: check_film_gamma.py). Same grid and network rebuild as check_film_gamma.py (doses 1-8 mg on a 0.25-mg grid,
normalised by 8 mg). Checkpoints whose maps receive z0 are listed and skipped. Writes results/review22/film_gamma_all.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import glob, os, sys
EP = os.environ.get('GAMMA_EP', '006000')   # Km confounding runs end at ep3000: GAMMA_EP=003000
import torch
os.chdir(REPO)
ROOTS = sys.argv[1:] or ['s4_pnoise_win', 's4_pnoise_win_more', 's4_pnoise_win_gpos', 'vcw200_matched', 'vcw400_matched', 'vcw_matched',
                         's4_nsweep2', 's4_pnoise', 's4_pnoise_decres', 's4_pnoise_dech0', 's4_pnoise_factonly', 's4_pnoise_sig001',
                         's4_pnoise_sig0217', 'km_matched', 'ablation_noOT', 's4_latent', 's4_orth']
grid = torch.arange(1.0, 8.0001, 0.25)
L = [__doc__.split('\n')[0], '']
for r in ROOTS:
    fs = sorted(glob.glob(f'results/{r}/**/traj/*film*ep{EP}.ckpt', recursive=True))
    mins, negs, skip = [], [], 0
    for f in fs:
        ck = torch.load(f, map_location='cpu', weights_only=False)
        sd = ck['state_dict'] if 'state_dict' in ck else ck
        a = ck.get('args'); a = vars(a) if hasattr(a, '__dict__') else (a or {})
        W = {k.split('film_gamma.')[1]: v for k, v in sd.items() if '.film_gamma.' in k or k.startswith('film_gamma.')}
        if not a.get('film_no_z0_cond', False) or '0.weight' not in W or '2.weight' not in W or '4.weight' in W:
            skip += 1; continue
        net = torch.nn.Sequential(torch.nn.Linear(W['0.weight'].shape[1], W['0.weight'].shape[0]), torch.nn.ReLU(),
                                  torch.nn.Linear(W['2.weight'].shape[1], W['2.weight'].shape[0]))
        net.load_state_dict(W)
        d1, d2 = torch.meshgrid(grid / 8.0, grid / 8.0, indexing='ij')
        c = torch.zeros(d1.numel(), W['0.weight'].shape[1]); c[:, 0] = d1.flatten(); c[:, 1] = d2.flatten()
        with torch.no_grad():
            g = net(c)
            if a.get('film_gamma_pos', False):          # as lib/latent_ode.py _film_maps: softplus takes precedence over the residual form
                g = torch.nn.functional.softplus(g)
            elif a.get('film_residual_dose', False):
                g = 1.0 + (c[:, 1:2] - c[:, 0:1]) * g
        mins.append(g.min().item()); negs.append((g <= 0).float().mean().item())
    if not fs:
        L.append(f'{r:22s} no ep{EP} OT-FiLM checkpoint'); continue
    if not mins:
        L.append(f'{r:22s} {len(fs)} checkpoints, all skipped (maps receive z0 or other architecture)'); continue
    nneg = sum(m <= 0 for m in mins)
    L.append(f'{r:22s} {len(mins):3d} ckpts (skipped {skip}): min gamma {min(mins):+.3f}; ckpts with some gamma <= 0: {nneg}; '
             f'max share <= 0 {100 * max(negs):.2f} %, mean share {100 * sum(negs) / len(negs):.3f} %')
open('results/review22/film_gamma_all.txt' if EP == '006000' else f'results/review22/film_gamma_all_ep{EP}.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
