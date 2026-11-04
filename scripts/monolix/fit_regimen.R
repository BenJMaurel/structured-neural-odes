# Population fit (SAEM, Monolix) of the training patients of Baaz et al.'s regimen test:
# "test a linear MAP-BE ... did you fit the model with SAEM or just assume the population parameters?").
#   Rscript scripts/monolix/fit_regimen.R <data.csv> <out_dir> <lin|mm> [arm covariate on: e.g. ke or ka,ke,V]
# (the arm covariate needs the ARM800 column: mapbe_pop.py export ... --arm-col; log(theta) = log(theta_pop) + beta*ARM800 + eta)
# ka, ke, V log-normal with a full correlation block (as the simulator), constant (additive) error; mm adds Km_pop without IIV.
# Starting values are generic (NOT the truth 0.4 / 0.6 / 50, Km 500). Writes estimates.json (estimates, log-likelihood).
suppressPackageStartupMessages({ library(lixoftConnectors); library(jsonlite) })
a <- commandArgs(trailingOnly = TRUE)
data_csv <- normalizePath(a[1]); out_dir <- a[2]; struct <- a[3]
arm_on <- if (length(a) >= 4) strsplit(a[4], ",")[[1]] else character(0)
stopifnot(struct %in% c("lin", "mm"))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE); out_dir <- normalizePath(out_dir)
model <- normalizePath(sprintf("scripts/monolix/model_regimen_%s.txt", struct))
t0 <- Sys.time(); say <- function(...) cat(sprintf("[%s] ", format(Sys.time(), "%H:%M:%S")), ..., "\n", sep = "")
invisible(initializeLixoftConnectors(software = "monolix",
  path = "/Applications/monolixSuite2024R1.app/Contents/Resources/monolixSuite", force = TRUE))
setPreferences(threads = 1)
has_arm <- "ARM800" %in% names(read.csv(data_csv, nrows = 1))
newProject(modelFile = model, data = list(dataFile = data_csv,
  headerTypes = c("id", "time", "amount", "observation", if (has_arm) "contcov")))
if (length(arm_on)) {
  stopifnot(has_arm)
  do.call(setCovariateModel, setNames(lapply(arm_on, function(p) c(ARM800 = TRUE)), arm_on))
}
obs_name <- names(getContinuousObservationModel()$errorModel)[1]
do.call(setErrorModel, setNames(list("constant"), obs_name))
setCorrelationBlocks(ID = list(c("ka", "ke", "V")))
setPopulationParameterInformation(ka_pop = list(initialValue = 1), ke_pop = list(initialValue = 0.3),
                                  V_pop = list(initialValue = 30))
if (struct == "mm") {
  setIndividualParameterVariability(Km = FALSE)
  setPopulationParameterInformation(Km_pop = list(initialValue = 300))
}
print(getPopulationParameterInformation()[, c("name", "initialValue", "method")])
proj <- file.path(out_dir, sprintf("regimen_%s.mlxtran", struct)); saveProject(projectFile = proj)
runPopulationParameterEstimation()
est <- getEstimatedPopulationParameters()
say("SAEM done after ", round(as.numeric(Sys.time() - t0, units = "mins"), 1), " min"); print(round(est, 5))
runConditionalDistributionSampling()
runLogLikelihoodEstimation(linearization = FALSE)
ll <- getEstimatedLogLikelihood(); print(ll)
saveProject(projectFile = proj)
write_json(list(data = data_csv, struct = struct, arm_covariate_on = arm_on, estimates = as.list(est), loglik = ll,
                minutes = as.numeric(Sys.time() - t0, units = "mins")),
           file.path(out_dir, "estimates.json"), auto_unbox = TRUE, digits = 10, pretty = TRUE)
say("wrote ", file.path(out_dir, "estimates.json")); cat("FIT_DONE\n")
