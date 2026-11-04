# Third-party code

All three projects below are under the MIT licence.

| Project | Where it is used here | Source |
|---|---|---|
| Latent ODEs for irregularly-sampled time series, Rubanova, Chen and Duvenaud | `lib/` and `paper_era/lib/` start from this code, and `run_models.py` from its main script | https://github.com/YuliaRubanova/latent_ode |
| Empirical Bayes variational autoencoder, Baaz et al. | `third_party/baaz_ebvae/`, with its own `LICENSE` | https://github.com/FraunhoferChalmersCentre/neuralODEpopulation |
| Neural-ODE for pharmacokinetics, Lu et al. | `scripts/regimen/lu_regimen.py` and `lib/lu_neural_pk.py` are our implementations, written from their papers and their public code | https://github.com/jameslu01/Neural_PK |

## `third_party/baaz_ebvae/`

* `lib/`, `scripts/` and `config.yaml` come from the public code of Baaz et al. Their script for clinical data is left out.
* `ours/` holds our drivers for the comparisons in the paper.
* `ours/PATCHES.md` explains our changes to their library. Each change is switched on by an environment variable, and with none
  set the library behaves as the original. `ours/changes_to_upstream.diff` is the exact difference.
