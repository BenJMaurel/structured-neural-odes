# Changes to the Baaz et al. (2026) public code for the PharmaNODE comparison

Original code: https://github.com/FraunhoferChalmersCentre/neuralODEpopulation (MIT), code files only, downloaded 2026-10-03.
Originals kept as `*.orig`. **Every change is opt-in through an environment variable; with none set the library is the
original** (checked: their own simulated design, seeded, 2 threads, gives line-identical training losses with the
patched and the original files).

| id | file | env var | what | why |
|---|---|---|---|---|
| P1 | lib/utils/utils_shared.py `prepare_ode_input` | `BAAZ_PER_DOSE=1` | each dose event keeps its own amount | the original gives every dose of a subject the FIRST dose's amount (`[:, :1].repeat`), so a within-patient dose change (d1 -> d2) cannot be represented |
| P2 | utils_preprocess.py `TrajectoryDataset`, utils_shared.py `encode_latent` | `BAAZ_COV_COLS=FORM,CYP,HTN` | per-subject vector of several (continuous) covariates | the original reads one categorical column (`COV_IND`) and keeps `cov[:, :1]`; encoder and prior already size their layers by `dim_cov` |
| P3 | utils_preprocess.py collate, utils_shared.py `preprocess_batch` | `BAAZ_ENC_COL=ENC` | the encoder sees exactly the flagged observations (and the loss all of them) | the original encoder sees all observations up to a single time cutoff; ours sees only the visit-1 samples at 0, 1, 3 h |

Not a code change: the dose-pulse width (`ODEFunc.log_sigma`, learnable) is initialised to the same PHYSICAL width as in
their design (0.05 of a 30-h horizon = 1.5 h), i.e. 1.5 / max_time on our timelines; the time grid has the same step
(0.25 h).

Evaluation: their `compute_test_metrics` cannot run with the shipped library (unpacks 14 of 13 values from
`preprocess_batch`; extra positional argument to `prepare_ode_input`); our drivers implement what it intends
(encoder on the visible samples, posterior mean, ODE prediction at the target times, de-standardised).
