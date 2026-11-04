#!/usr/bin/env python3
"""Where does variant A keep the patient information that sets exposure?

The physiology-only tracker (track_null.py --phys-only) barely detects real visit-2 Vmax changes. This probes why: test
patients' visit-1 encoder means are transported to 4 mg by A's own FiLM, written in A's basis (FiLM 1 / dynamics 5 /
physiology 4 coordinates), and each block is probed (ridge, 5-fold CV R^2) for the true log Vmax, log Km, log Vc, log Ktr,
the true log AUC at 4 mg and the MODEL's own log AUC at 4 mg (what the tracker's readout g sees).

    phys_probe.py [seeds ...] > results/tracking/phys_probe.txt
"""
import os, sys
import numpy as np, pandas as pd, torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
sys.path.insert(0, os.getcwd())
from lib.read_tacro import extract_gen_tac_film, TacroFilmDataset, collate_fn_tacro_film, set_static_hematocrit
from probe_entanglement import load_film_model

T = 'confound_vc00_s4_pnoise_win_n100'
TAG = 'noz0_sig0.05_sc141.27_dech50_sel-mse_v2_decres'
COHORT = 'confound_vc00_s4_pnoise_win'
TRUTH = 'results/exp_film_run/confound_vc00_s4_pnoise_win_shift70/confound_truth.csv'   # same patients, visit-1 params


def r2(X, y):
    m = RidgeCV(alphas=np.logspace(-4, 3, 15))
    p = cross_val_predict(m, X, y, cv=KFold(5, shuffle=True, random_state=0))
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    seeds = [int(s) for s in sys.argv[1:]] or [1, 2, 3]
    torch.set_num_threads(1)
    ck0 = (f'results/s4_orth/osplit1-5_n100_s{seeds[0]}/exp_film_run/{T}/traj/experiment_film_{T}__{TAG}_osplit1-5_ep006000.ckpt')
    SD4 = int(getattr(torch.load(ck0, map_location='cpu', weights_only=False)['args'], 'static_dim', 3)) >= 4
    set_static_hematocrit(SD4)          # the loader fixes the static width at load time
    pin = extract_gen_tac_film(file_path=[f'results/exp_film_run/{T}/virtual_cohort_film_train.csv'])[1]
    MAX_OUT, LAM, DMAX = float(pin[0]), float(pin[1]), float(pin[2])
    tr, te = (f'results/exp_film_run/{COHORT}/virtual_cohort_film_{s}.csv' for s in ('train', 'test'))
    data, _ = extract_gen_tac_film(file_path=[tr, te], scale=pin)
    train_ids = set(extract_gen_tac_film(file_path=[tr], scale=pin)[0].keys())
    ids = [k for k in data if k not in train_ids]
    truth = pd.read_csv(TRUTH)
    truth['ID'] = truth['ID'].astype(int).astype(str)
    truth = truth.set_index('ID').loc[[str(int(float(i))) for i in ids]]
    targets = {f'log {c}': np.log(truth[c].values.astype(float)) for c in ('Vmax', 'Km', 'Vc', 'Ktr')}
    targets['true log AUC_v1'] = np.log(truth['AUC_v1'].values.astype(float))
    print(__doc__.strip().split('\n\n')[1].replace('\n', ' '))
    print(f'\n{len(ids)} test patients of {COHORT}; ridge 5-fold CV R^2 per block (dims in brackets)')
    rows = {}
    for s in seeds:
        ck = (f'results/s4_orth/osplit1-5_n100_s{s}/exp_film_run/{T}/traj/experiment_film_{T}__{TAG}_osplit1-5_ep006000.ckpt')
        a0 = torch.load(ck, map_location='cpu', weights_only=False)['args']
        assert (int(getattr(a0, 'static_dim', 3)) >= 4) == SD4
        model, args = load_film_model(ck, torch.device('cpu'))
        ds = TacroFilmDataset({k: data[k] for k in ids})
        b = collate_fn_tacro_film([ds[i] for i in range(len(ids))], torch.device('cpu'))
        with torch.no_grad():
            enc_in, st = model._encoder_input(b['observed_data_v1'], b['dose_v1'], b['static_v1'])
            mu, _ = model.encoder_z0(enc_in, b['observed_tp_v1'], static=st, run_backwards=True)
            z = mu[0]
            d1 = b['dose_v1'].float(); dref = torch.full_like(d1, 4.0 / DMAX)
            c = model._film_context(torch.stack([d1, dref], 1).unsqueeze(0), z.unsqueeze(0), None)
            gamma, beta = model._film_maps(c, d1, dref, 1)
            zr = model._film_apply(z.unsqueeze(0), gamma, beta)[0]
            tp = torch.linspace(0, 24, 100)
            m12 = ((tp[1:] <= 12.0) & (tp[:-1] <= 12.0)).float()
            form = b['static_v1'][:, 1].float()

            def logauc(zz):
                pred = model.decoder(model.diffeq_solver(zz.unsqueeze(0), tp))[0, :, :, 0]
                base = LAM * pred + 1.0
                conc = torch.where(base > 0, base.clamp(min=1e-12) ** (1.0 / LAM), torch.zeros_like(base)) * MAX_OUT
                seg = 0.5 * (conc[:, 1:] + conc[:, :-1]) * (tp[1:] - tp[:-1])
                return torch.log(torch.where(form > 0.5, (seg * m12).sum(1), seg.sum(1))).numpy().astype(float)
            auc = np.exp(logauc(zr))
            Q = model.orth_split.Q()
            kf, kd = model.orth_split.k_f, model.orth_split.k_d
            C = (zr @ Q).numpy().astype(float)
        blocks = {f'FiLM [{kf}]': C[:, :kf], f'dynamics [{kd}]': C[:, kf:kf + kd], f'physiology [{C.shape[1] - kf - kd}]': C[:, kf + kd:],
                  f'non-physiology [{kf + kd}]': C[:, :kf + kd], f'full z [{C.shape[1]}]': C}
        tg = dict(targets); tg['MODEL log AUC at 4 mg'] = np.log(auc)
        var = {k: float(np.var(v, axis=0).sum()) for k, v in blocks.items()}
        for bk, X in blocks.items():
            for tk, y in tg.items():
                rows.setdefault((tk, bk), []).append(r2(X, y))
        rows.setdefault(('_var', 'x'), []).append(var)
        # INTERVENTIONAL check: give every patient another patient's coordinates in ONE block (others kept) and re-decode.
        # share = E (g_swap - g)^2 / (2 Var g): ~ the fraction of the model's exposure variance that block actually drives
        # (1 = all of it, 0 = the decoder ignores the block even if the block is correlated with exposure).
        g0 = np.log(auc); perm = np.random.default_rng(s).permutation(len(ids)); Qt = Q.t()
        sl = {'FiLM': slice(0, kf), 'dynamics': slice(kf, kf + kd), 'physiology': slice(kf + kd, C.shape[1]), 'non-physiology': slice(0, kf + kd)}
        for name, idx in sl.items():
            Cs = C.copy(); Cs[:, idx] = C[perm, idx]
            with torch.no_grad():
                gs = logauc(torch.as_tensor(Cs, dtype=torch.float32) @ Qt)
            rows.setdefault(('swap', name), []).append((float(np.mean((gs - g0) ** 2) / (2 * np.var(g0))), 100 * float(np.median(np.abs(np.expm1(gs - g0))))))
        print(f'  seed {s}: spread (total variance) per block ' + ', '.join(f'{k} {v:.3f}' for k, v in var.items()))
    bks = list(blocks)
    print('\n' + f'{"target":26s}' + ''.join(k.rjust(20) for k in bks))
    for tk in tg:
        print(f'{tk:26s}' + ''.join((f'{np.mean(rows[(tk, bk)]):.2f} [{min(rows[(tk, bk)]):.2f}-{max(rows[(tk, bk)]):.2f}]').rjust(20)
                                    for bk in bks))

    print('\nINTERVENTION (swap one block between patients, keep the rest, re-decode the model log AUC at 4 mg):')
    for name in ('FiLM', 'dynamics', 'physiology', 'non-physiology'):
        v = rows[('swap', name)]
        print(f'  swap {name:15s} share of exposure variance {np.mean([x[0] for x in v]):.2f} [{min(x[0] for x in v):.2f}-{max(x[0] for x in v):.2f}]'
              f'   median |AUC change| {np.mean([x[1] for x in v]):.0f} %')


if __name__ == '__main__':
    main()
