#!/usr/bin/env python3
"""Is the OT-FiLM scale gamma(d1, d2) positive in every trained Table 2 model? (review 22/10)
An affine map z -> gamma*z + beta is the W2 optimal-transport map between diagonal Gaussians only if gamma > 0 elementwise.
The maps receive (d1, d2) only (--film-no-z0-cond: the z0 block of the context is zeroed), so gamma is a function of the dose
pair. Evaluated on a 0.25-mg grid of 1-8 mg (doses normalised by the training maximum, read from the checkpoint args / 8 mg).
Writes results/review22/film_gamma.txt"""
import os as _os; REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..'))
import glob, os, sys
import torch
os.chdir(REPO)
os.makedirs('results/review22', exist_ok=True)
ROOTS = {s: ('results/s4_pnoise_win' if s <= 3 else 'results/s4_pnoise_win_more') for s in range(1, 11)}
L = [__doc__.split('\n')[0], '']
grid = torch.arange(1.0, 8.0001, 0.25)
allmin = []
for s, root in ROOTS.items():
    f = glob.glob(f'{root}/film_n100_s{s}/**/traj/*ep006000.ckpt', recursive=True)
    assert len(f) == 1, (s, f)
    ck = torch.load(f[0], map_location='cpu', weights_only=False)
    sd = ck['state_dict'] if 'state_dict' in ck else ck
    a = ck.get('args'); a = vars(a) if hasattr(a, '__dict__') else (a or {})
    assert a.get('film_no_z0_cond', True), 'model conditions on z0'
    resid = bool(a.get('film_residual_dose', False))
    W = {k.split('film_gamma.')[1]: v for k, v in sd.items() if '.film_gamma.' in k or k.startswith('film_gamma.')}
    net = torch.nn.Sequential(torch.nn.Linear(W['0.weight'].shape[1], W['0.weight'].shape[0]), torch.nn.ReLU(),
                              torch.nn.Linear(W['2.weight'].shape[1], W['2.weight'].shape[0]))
    net.load_state_dict(W)
    din = W['0.weight'].shape[1]
    d1, d2 = torch.meshgrid(grid / 8.0, grid / 8.0, indexing='ij')
    c = torch.zeros(d1.numel(), din); c[:, 0] = d1.flatten(); c[:, 1] = d2.flatten()
    with torch.no_grad():
        g = net(c)
        if resid:
            g = 1.0 + (c[:, 1:2] - c[:, 0:1]) * g
    gmin = g.min().item(); allmin.append(gmin)
    neg = (g <= 0).float().mean().item()
    L.append(f'seed {s:2d}: gamma min {gmin:+.3f}  max {g.max().item():+.3f}  share of (pair, coordinate) <= 0: {100 * neg:.2f} %'
             f'  residual-dose param: {resid}  ({os.path.basename(f[0])[:60]}...)')
L += ['', f'overall min gamma over 10 seeds: {min(allmin):+.4f}  -> ' + ('POSITIVE everywhere' if min(allmin) > 0 else 'NOT positive')]
open('results/review22/film_gamma.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
