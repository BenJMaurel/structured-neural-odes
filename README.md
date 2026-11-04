# Learning pharmacokinetic dynamics with structured neural ODEs for precision dosing

Code for the simulated experiments of the paper by Maurel, Guilloux, Zohar, Ursino and Woillard. It covers the models, the
simulators, the baselines and the tables and figures that come from simulated patients.

**Not included:** the clinical data, and the code that reads them. These data cannot be shared (see the Data availability
statement of the paper), so the results on kidney transplant recipients (the clinical cohorts at the observed dose and the next
monitoring visit, Table 4) cannot be reproduced from this repository. No data, checkpoints or results are stored here. Every
cohort is simulated by the scripts below.

## Setup

Python 3.12, CPU only. The versions we used are pinned in `requirements.txt`.

```bash
pip install -r requirements.txt
bash scripts/smoke_test.sh
```

`scripts/smoke_test.sh` runs each main step for a few iterations on a small cohort, to check the installation. It takes a few
minutes and its numbers mean nothing.

The Bayesian estimation rows also need R with `lixoftConnectors` (Monolix 2024R1), `mapbayr` 0.10.2, `mrgsolve`, `tidyverse`,
`glue`, `argparse`, `furrr` and `MESS`. Set `MLX` to your Monolix folder if it is not in the default macOS location.

## Running the experiments

Run everything from the repository root, with one thread per job (`export OMP_NUM_THREADS=1`). The full experiments are many
small runs and take hours to days.

The scripts named `chain_*` and `queue_*` are the commands as we ran them.

* They skip a job whose output already exists, so they can be restarted.
* They start a job only when few others are running. The limit is written near the top of each script.
* Some of them wait for a marker written by an earlier queue. Start them with `SKIP_WAIT=1` to run them without waiting.
* Set `PYTHON` if `python3` is not the interpreter you want.

`run_models.py` and `train_dose_cond.py` expect the checkpoint folder to exist, for example `mkdir -p <save>/exp_film_run/<experiment>`.
The chain scripts create it.

Outputs go to `results/`, `out/`, `logs/` and `figures/`, which git ignores. The folders also hold exploratory runs that the
paper does not use. The table below lists what the paper uses.

## From the paper to the code

| In the paper | Cohorts | Runs | Table or figure |
|---|---|---|---|
| Table 1, scenarios 1 to 3 | `scripts/scen_rerun/gen_seeded.py` | `scripts/scen_rerun/setup_slots.sh`, then `scripts/scen_rerun/chain_scen_rerun.sh` (one run: `scripts/scen_rerun/run_one.sh`) | `scripts/scen_rerun/summarise.py` |
| Table 2, Supplementary Table 1 | `scripts/s4/gen_window_all.sh`, `gen_full_holdout.sh`, `gen_s4_ood.sh` | `scripts/s4/chain_table1_window.sh`, `scripts/weekend/queue_weekend.sh`, `scripts/review22/queue_review22.sh` | `scripts/ndm/ndm_tables.py`, `scripts/weekend/table2_more.py`, `scripts/review22/table2_gpos.py` |
| Bayesian estimation rows of Table 2 | as above | `scripts/monolix/run_monolix_chain.sh`, `scripts/monolix/chain_lin_s4.sh`, `scripts/review22/queue_baselines.sh` | `scripts/review22/cf_baselines_table.py` |
| Dose response of the two-port model | as above | `scripts/s4/chain_table1_window.sh` | `scripts/ndm/lu_dose_elasticity.py` |
| Dose-switch cohort, Supplementary Note 2 | `scripts/baaz/gen_switch_cohorts.sh` | `scripts/baaz/chain_part2.sh`, `scripts/baaz/chain_part2b.sh` | `scripts/baaz/part2_table.py` |
| Table 3a, Supplementary Table 5 | `third_party/baaz_ebvae/ours/regimen/gen_regimens.py` | `scripts/weekend/queue_weekend.sh`, `scripts/review22/queue_baselines.sh`, `scripts/regimen/queue_lu.sh` | `scripts/weekend/confirm_table.py`, `scripts/review22/regimen_baselines_table.py`, `scripts/review22/lu_protocol_table.py` |
| Supplementary Table 2, development runs | as above | `scripts/regimen/queue_claim5.sh`, `scripts/regimen/queue_bfix.sh`, `scripts/regimen/queue_dc_tune.sh`, `scripts/regimen/queue_dc_phase2.sh` | `scripts/regimen/freenn_ablation_table.py`, `scripts/regimen/table4.py` |
| Table 3b, Supplementary Table 4, Fig. 3 | `scripts/regimen/gen_tacro_regimens.py` | `scripts/regimen/queue_tacro_visit.sh`, `scripts/regimen/queue_tacro_s2.sh`, `scripts/regimen/queue_tacro_s2_ebvae.sh`, `scripts/regimen/queue_tacro_s2_lu.sh`, `scripts/review22/queue_review22.sh`, `scripts/review22/clip0_visit1.sh` | `scripts/review22/visit1_table.py`, `scripts/regimen/tacro_design_table.py`, `scripts/regimen/clip0_eval.py`, `scripts/ndm/fig_tacro_patients_data.py`, `scripts/ndm/pass3/make_fig_tacro_patients.py` |
| Titrated arms, Fig. 4 | as Table 3a | `scripts/weekend/queue_weekend.sh`, `scripts/regimen/queue_armbal.sh` | `scripts/weekend/arm_split_all.py`, `scripts/regimen/baaz_arm_split.py`, `scripts/ndm/pass3/make_fig_ebvae_vs_ours.py` |
| Removing the dose from the prior | as Table 3a | `scripts/regimen/queue_eb_priortest.sh`, `scripts/regimen/queue_eb_priortest_baaz.sh` | `scripts/regimen/eb_priortest_table.py`, `scripts/regimen/eb_priortest_baaz_table.py` |
| Empirical Bayes model on Scenario 3 | `third_party/baaz_ebvae/ours/tacro/convert_scen.py` | `scripts/baaz/queue_ebconv.sh`, `scripts/baaz/queue_ebcheck.sh`, `scripts/baaz/queue_ebcheck2.sh`, `scripts/baaz/run_mapref.sh`, `scripts/baaz/run_lode_split.sh` | `scripts/baaz/ebconv_table.py`, `scripts/baaz/ebcheck_table.py`, `scripts/baaz/gap_table.py` |
| Doses titrated on the Michaelis constant or the central volume | `gen_tacro_confound.py`, `scripts/nsweep/make_subsets.py` | `scripts/confound_eval/chain_km_matched.sh`, `scripts/confound_eval/chain_vcw200_matched.sh`, `scripts/confound_eval/chain_vcw_matched.sh`, `scripts/confound_eval/queue_vcw_seeds.sh` | `scripts/confound_eval/km_matched_did.py`, see `scripts/confound_eval/README.md` |
| Variance of the dynamics due to the dose | as above | as above | `scripts/vc/field_sensitivity.py` |
| Conserved-physiology variant | as Table 2 | `scripts/s4/chain_orth.sh`, `scripts/tracking/queue_null.sh`, `scripts/tracking/queue_recovery.sh`, `scripts/tracking/queue_aphys.sh` | `scripts/s4/orth_table.py`, `scripts/tracking/phys_table.py`, `scripts/tracking/phys_probe.py` |
| Supplementary Fig. 2 | as Table 2 | as Table 2 | `scripts/ndm/fig_transport_at_t_data.py`, `scripts/ndm/fig_transport_history_data.py`, `scripts/ndm/pass3/make_fig_transport_history.py` |
| Schematic figures 1, 2, 5 and 6 | none | none | `scripts/ndm/pass3/make_fig_overview.py`, `scripts/ndm/pass3/make_fig_designs.py`, `scripts/ndm/make_fig_otfilm_arch.py`, `scripts/ndm/make_fig_dose_event_arch.py` |

**About Table 1.** `scripts/scen_rerun/` is the rerun on 100 datasets described in the Methods of the paper. In Table 1, only the
latent ODE RMSPE of Scenario 3 comes from this rerun. The other cells come from an earlier analysis, so this pipeline will give
values that differ from them. It runs the code of the earlier pipeline, kept as it was in `paper_era/`.

## The models

| In the paper | Training | Evaluation |
|---|---|---|
| Plain latent ODE | `run_models.py --latent-ode` | `scripts/repro/eval_lode.py` |
| Dose-shift model (OT-FiLM) | `run_models.py --latent-ode --use_film`, flags in `scripts/s4/chain_table1_window.sh` | `test_film_matched.py` |
| Dose-in-the-dynamics model | `train_dose_cond.py` | `test_dose_cond.py` |
| Dose-event model, two-visit cohort | `scripts/s4/chain_doseh.sh` | `scripts/s4/doseh_table.py` |
| Dose-event model, regimen designs | `scripts/regimen/dh_regimen.py`, with `--obsrv-mgl 20 --kick-phys 32 --enc-dose norm` | written by the training script |
| Two-port model of Lu et al. | `train_lu_pk.py`, `scripts/regimen/lu_regimen.py nmi` | `test_lu_pk.py` |
| Regimen model of Lu et al. | `scripts/regimen/lu_regimen.py is` | written by the training script |
| Empirical Bayes model of Baaz et al. | `third_party/baaz_ebvae/ours/regimen/run_regimen_titrated.py`, `third_party/baaz_ebvae/ours/regimen/run_regimen_tacro.py`, `third_party/baaz_ebvae/ours/tacro/run_tacro_baaz.py` | written by the training script |

Simulators: `gen_tacro_confound.py` (two-visit and dose-switch cohorts), `scripts/regimen/gen_tacro_regimens.py` (tacrolimus regimen
design), `third_party/baaz_ebvae/ours/regimen/gen_regimens.py` (design of Baaz et al.), `paper_era/gen_tacro.py` (scenarios 1 to 3).

## Pitfalls that give wrong numbers without any error

1. **Pass `--scale-from <training cohort>`** to `test_film_matched.py` and `test_dose_cond.py`. Without it, the normalisation is
   refitted on the evaluated cohort.
2. **Do not use `Test MSE` from the training logs or the `_best` checkpoints**, apart from reproducing Table 1, as its Methods
   section explains. The training loop refits the test-split scaler, which corrupts this signal. The paper reads fixed-epoch
   checkpoints from `traj/`.
3. **Prefer the median error to the raw RMSPE of the AUC** on simulated cohorts. A few patients with a small true AUC dominate it.
4. **Checkpoint names encode the configuration, not the number of epochs.** Two runs of one configuration under the same `--save`
   folder overwrite each other.
5. **Compare models with the same likelihood and readout.** `train_dose_cond.py` falls back to `--noise-weight` when `--obsrv-std`
   is not set, and its default readout differs from that of the dose-shift model.
6. **Pass `--prop-sd 0.113 --add-sd 0.71`** to `gen_tacro_confound.py` for the published residual error. `gen_tacro_film.py` has a
   lower residual error.

## Licence and third-party code

MIT, see `LICENSE`. `THIRD_PARTY.md` lists the code this repository builds on.
