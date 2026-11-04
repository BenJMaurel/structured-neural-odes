# Log-likelihood of an EXISTING Monolix fit of the 3-scenario rerun (2026-10-02): is the degenerate solution reached
# from default / auto-init a worse local optimum, or the better fit of the misspecified model?
# Works on a COPY of the fit project (the caller copies it): loads the project, sets every population parameter to
# its estimate from populationParameters.txt and FIXES it, runs SAEM (nothing to estimate: only the individual
# parameters are sampled), the conditional distribution, then -2LL by importance sampling (seed fixed).
#   Rscript loglik_fit.r <project.mlxtran (copy)> <populationParameters.txt> <out.json>
suppressMessages({library(lixoftConnectors); library(jsonlite)})
a <- commandArgs(trailingOnly = TRUE); prj <- a[1]; ppf <- a[2]; out <- a[3]
initializeLixoftConnectors(software = "monolix",
  path = Sys.getenv("MLX_PATH", "/Applications/MonolixSuite2024R1.app/Contents/Resources/monolixSuite"))
setPreferences(threads = as.integer(Sys.getenv("MLX_THREADS", "1")))
loadProject(prj)
pp <- read.csv(ppf); est <- setNames(pp$value, pp$parameter)
info <- getPopulationParameterInformation()
miss <- setdiff(info$name[info$method != "FIXED"], names(est))
if (length(miss)) stop("no estimate for: ", paste(miss, collapse = ", "))
k <- info$name %in% names(est)
info$initialValue[k] <- est[info$name[k]]; info$method[k] <- "FIXED"
setPopulationParameterInformation(info)
setProjectSettings(seed = 123456)
runPopulationParameterEstimation()
runConditionalDistributionSampling()
runLogLikelihoodEstimation(linearization = FALSE)
ll <- getEstimatedLogLikelihood()
print(ll)
v <- ll[["importanceSampling"]]
res <- list(project = prj, params = as.list(est), ofv = unname(v[["OFV"]]), aic = unname(v[["AIC"]]), bic = unname(v[["BIC"]]),
            ll_all = lapply(ll, function(x) as.list(x)))
write_json(res, out, auto_unbox = TRUE, digits = NA, pretty = TRUE)
cat("LOGLIK_DONE", out, "OFV =", res$ofv, "\n")
