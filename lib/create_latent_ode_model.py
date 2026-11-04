###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Modified version of Yulia Rubanova
###########################

import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import relu

from . import utils
from .latent_ode import LatentODE, LatentODEGMM, LatentODEGMM_V, LatentODEFlow
from .encoder_decoder import *
from .diffeq_solver import DiffeqSolver
from .ode_func import ODEFunc, ODEFunc_w_Poisson

from torch.distributions.normal import Normal


#####################################################################################################

def create_LatentODE_model(args, input_dim, z0_prior, obsrv_std, device, 
	classif_per_tp = False, n_labels = 1):

	dim = args.latents
	if args.poisson:
		lambda_net = utils.create_net(dim, input_dim, 
			n_layers = 1, n_units = args.units, nonlinear = nn.Tanh)

		# ODE function produces the gradient for latent state and for poisson rate
		ode_func_net = utils.create_net(dim * 2, args.latents * 2, 
			n_layers = args.gen_layers, n_units = args.units, nonlinear = nn.Tanh)

		gen_ode_func = ODEFunc_w_Poisson(
			input_dim = input_dim, 
			latent_dim = args.latents * 2,
			ode_func_net = ode_func_net,
			lambda_net = lambda_net,
			device = device).to(device)
	else:
		dim = args.latents 
		ode_func_net = utils.create_net(dim, args.latents, 
			n_layers = args.gen_layers, n_units = args.units, nonlinear = nn.Tanh)

		gen_ode_func = ODEFunc(
			input_dim = input_dim, 
			latent_dim = args.latents, 
			ode_func_net = ode_func_net,
			device = device).to(device)

	z0_diffeq_solver = None
	n_rec_dims = args.rec_dims
	# enc_input_dim = (int(input_dim) + 3) * 2 # we add the static values then concatenate the mask
	enc_input_dim = int(input_dim) + 1 # we concatenate the dose and other (dynamic) covariates

	gen_data_dim = input_dim

	z0_dim = args.latents
	if args.poisson:
		z0_dim += args.latents # predict the initial poisson rate

	if args.z0_encoder == "odernn":
		ode_func_net = utils.create_net(n_rec_dims, n_rec_dims, 
			n_layers = args.rec_layers, n_units = args.units, nonlinear = nn.Tanh)

		rec_ode_func = ODEFunc(
			input_dim = enc_input_dim, 
			latent_dim = n_rec_dims,
			ode_func_net = ode_func_net,
			device = device).to(device)

		z0_diffeq_solver = DiffeqSolver(enc_input_dim, rec_ode_func, "euler", args.latents, 
			odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)
		
		encoder_z0 = Encoder_z0_ODE_RNN(n_rec_dims, enc_input_dim, z0_diffeq_solver,
			z0_dim = z0_dim, n_gru_units = args.gru_units, device = device,
			static_dim = int(getattr(args, 'static_dim', 3))).to(device)

	elif args.z0_encoder == "rnn":
		encoder_z0 = Encoder_z0_RNN(z0_dim, enc_input_dim,
			lstm_output_size = n_rec_dims, device = device).to(device)
	else:
		raise Exception("Unknown encoder for Latent ODE model: " + args.z0_encoder)

	decoder = Decoder(args.latents, gen_data_dim, hidden_dim=getattr(args, 'decoder_hidden', 0),
		residual=getattr(args, 'decoder_residual', False)).to(device)

	diffeq_solver = DiffeqSolver(gen_data_dim, gen_ode_func, 'dopri5', args.latents, 
		odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

	use_gmm = getattr(args, 'use_gmm', False) 
	use_gmm_v = getattr(args, 'use_gmm_v', False) 
	use_flow = getattr(args, 'use_flow', False) 
	if use_gmm:
		n_components = getattr(args, 'n_components', 5)
		print(f"Initializing LatentODEGMM with {n_components} clusters.")

		model = LatentODEGMM(
		    input_dim = gen_data_dim, 
		    latent_dim = args.latents, 
		    encoder_z0 = encoder_z0, 
		    decoder = decoder, 
		    diffeq_solver = diffeq_solver, 
		    z0_prior = z0_prior, 
		    device = device,
		    n_components = n_components,  # Pass the number of clusters
		    obsrv_std = obsrv_std,
		    use_poisson_proc = args.poisson, 
		    use_binary_classif = args.classif,
		    linear_classifier = args.linear_classif,
		    classif_per_tp = classif_per_tp,
		    n_labels = n_labels,
		    train_classif_w_reconstr = (args.dataset == "physionet")
		    ).to(device)
		
	elif use_gmm_v:
		n_components = getattr(args, 'n_components', 5)
		print(f"Initializing LatentODEGMM with {n_components} clusters.")

		model = LatentODEGMM_V(
		    input_dim = gen_data_dim, 
		    latent_dim = args.latents, 
		    encoder_z0 = encoder_z0, 
		    decoder = decoder, 
		    diffeq_solver = diffeq_solver, 
		    z0_prior = z0_prior, 
		    device = device,
		    n_components = n_components,  # Pass the number of clusters
		    obsrv_std = obsrv_std,
		    use_poisson_proc = args.poisson, 
		    use_binary_classif = args.classif,
		    linear_classifier = args.linear_classif,
		    classif_per_tp = classif_per_tp,
		    n_labels = n_labels,
		    train_classif_w_reconstr = (args.dataset == "physionet")
		    ).to(device)
	elif use_flow:
		print(f"Initializing LatentODEFlow")

		model = LatentODEFlow(
		    input_dim = gen_data_dim, 
		    latent_dim = args.latents, 
		    encoder_z0 = encoder_z0, 
		    decoder = decoder, 
		    diffeq_solver = diffeq_solver, 
		    z0_prior = z0_prior, 
		    device = device,
		    obsrv_std = obsrv_std,
		    use_poisson_proc = args.poisson, 
		    use_binary_classif = args.classif,
		    linear_classifier = args.linear_classif,
		    classif_per_tp = classif_per_tp,
		    n_labels = n_labels,
		    train_classif_w_reconstr = (args.dataset == "physionet")
		    ).to(device)
	else:
        # Standard Model
		model = LatentODE(
		    input_dim = gen_data_dim, 
		    latent_dim = args.latents, 
		    encoder_z0 = encoder_z0, 
		    decoder = decoder, 
		    diffeq_solver = diffeq_solver, 
		    z0_prior = z0_prior, 
		    device = device,
		    obsrv_std = obsrv_std,
		    use_poisson_proc = args.poisson, 
		    use_binary_classif = args.classif,
		    linear_classifier = args.linear_classif,
		    classif_per_tp = classif_per_tp,
			film = args.use_film,
			film_occ = int(getattr(args, 'film_occ', 0)),
		    n_labels = n_labels,
		    train_classif_w_reconstr = (args.dataset == "physionet")
		    ).to(device)

	# --orth-split k_f,k_d: FiLM / dynamics / physiology on mutually orthogonal learned subspaces.
	# Only the GENERATIVE flow is projected; the encoder's ODE-RNN is untouched.
	if getattr(args, 'orth_split', None):
		from .latent_ode import OrthSplit
		k_f, k_d = (int(v) for v in str(args.orth_split).split(','))
		model.orth_split = OrthSplit(args.latents, k_f, k_d).to(device)
		model.diffeq_solver.ode_func._proj_fn = model.orth_split.P_dyn
		print(f"orth-split: FiLM {k_f} / dynamics {k_d} / physiology {args.latents - k_f - k_d} dims (learned orthonormal basis)")

	# --dose-history N (variant B): no FiLM transport; each visit is simulated through its dosing history, doses being
	# translations inside the drug subspace (FiLM + dynamics columns). The flow moves to that whole subspace, is zero at
	# drug-free states and gets a linear elimination term (LatentODE._dose_history_traj, ODEFunc.get_ode_gradient_nn).
	if int(getattr(args, 'dose_history', 0) or 0) > 0:
		from .latent_ode import DoseHistory
		if getattr(model, 'orth_split', None) is None:
			raise ValueError("--dose-history needs --orth-split k_f,k_d")
		model.dose_hist = DoseHistory(model.orth_split, args.dose_history, getattr(args, 'dose_interval', 24.0),
		                              getattr(args, 'dose_interval_prograf', 0.0) or 0.0).to(device)
		ode = model.diffeq_solver.ode_func
		ode._proj_fn = model.orth_split.P_drug
		ode._phys_fn = model.orth_split.P_phys
		ode._decay_fn = model.dose_hist.decay
		# Zero output layer: the flow starts as the pure linear elimination. With the default random init the drug-free
		# equilibrium is already unstable (max Re eig of the drug block +0.19 vs kappa 0.10, checked 2026-10-03) and after
		# 7 doses every patient sits at the same drug state (norm 22.7 for 1, 4 and 8 mg): the dose would start inert.
		nn.init.zeros_(ode.gradient_net[-1].weight)
		nn.init.zeros_(ode.gradient_net[-1].bias)
		print(f"dose-history: {model.dose_hist.n_doses} doses every {model.dose_hist.interval:g} h from the drug-free state; "
		      f"drug subspace {model.orth_split.k_f + model.orth_split.k_d} dims, no FiLM transport")
		_kp = int(getattr(args, 'kick_phys', 0) or 0)
		if _kp > 0:   # --kick-phys H (2026-10-12): jump b(p), last layer zero (starts at the population jump); own RNG fork
			_k = model.orth_split.k_f + model.orth_split.k_d
			with torch.random.fork_rng():
				torch.manual_seed(int(getattr(args, 'seed', 0) or 0) + 31)
				model.dose_hist.kick_net = nn.Sequential(nn.Linear(args.latents - _k, _kp), nn.Tanh(), nn.Linear(_kp, _k)).to(device)
			nn.init.zeros_(model.dose_hist.kick_net[-1].weight); nn.init.zeros_(model.dose_hist.kick_net[-1].bias)
			print(f"dose-history: physiology-dependent jump b(p), hidden {_kp}")
	elif int(getattr(args, 'kick_phys', 0) or 0) > 0:
		raise ValueError("--kick-phys needs --dose-history")
	if getattr(args, 'encoder_dose_norm', False):
		# --encoder-dose-norm (2026-10-12): Box-Cox lambda of the training cohort, filled by run_models.py, restored from checkpoints
		model.encoder_dose_norm = True
		model.register_buffer('enc_bc_lambda', torch.tensor(float('nan'), device=device))
		if getattr(args, 'encoder_dose_value', False):
			# --encoder-dose-value (2026-10-15): the encoder also reads the dose value through its dose channel
			# (last input of the GRU's three input layers) and static[:,0] (first input of the static encoder). Their weights start
			# at zero, so the initial encoder is exactly the dose-normalised one; no random numbers drawn.
			enc = model.encoder_z0
			assert isinstance(enc, Encoder_z0_ODE_RNN), "--encoder-dose-value needs the ODE-RNN encoder"
			with torch.no_grad():
				for lin in (enc.GRU_update.update_gate[0], enc.GRU_update.reset_gate[0], enc.GRU_update.new_state_net[0]):
					lin.weight[:, -1] = 0.0
				enc.static_encoder[0].weight[:, 0] = 0.0
			model.encoder_dose_value = True
	elif getattr(args, 'encoder_dose_value', False):
		raise ValueError("--encoder-dose-value needs --encoder-dose-norm")
	# --bio-match (2026-10-06): similarity-matched stand-in counterfactuals (LatentODE._bio_match_loss).
	if float(getattr(args, 'bio_match', 0.0) or 0.0) > 0:
		if getattr(model, 'dose_hist', None) is None:
			raise ValueError("--bio-match needs --dose-history")
		model.bio_match_coef = float(args.bio_match)
		model.bio_match_h = float(getattr(args, 'bio_match_h', 0.3))
		print(f"bio-match: weight {model.bio_match_coef:g}, kernel bandwidth {model.bio_match_h:g} (Box-Cox units) on the "
		      f"3 observed samples, neighbours with the same dose and formulation")
	# --bio-change gauss (2026-10-06): per-patient change of physiology between the two visits (LatentODE BioChange).
	# The training patients' IDs travel in args (set by run_models), so a checkpoint rebuilds the same table.
	if str(getattr(args, 'bio_change', 'none') or 'none') != 'none':
		from .latent_ode import BioChange
		if getattr(model, 'dose_hist', None) is None:
			raise ValueError("--bio-change needs --dose-history")
		ids = getattr(args, 'bio_change_ids', None)
		if not ids:
			raise ValueError("--bio-change: args.bio_change_ids (the training patient IDs) is missing")
		n_phys = args.latents - model.orth_split.k_f - model.orth_split.k_d
		model.bio_change = BioChange(n_phys, ids, tau0=getattr(args, 'bio_change_tau0', 0.3)).to(device)
		model.bio_change_orth_coef = float(getattr(args, 'bio_change_orth', 0.0) or 0.0)
		print(f"bio-change: per-patient change of the {n_phys} physiology dims between visits for {len(ids)} training "
		      f"patients, prior N(0, tau^2), tau learned from {getattr(args, 'bio_change_tau0', 0.3):g}"
		      + (f"; dose-orthogonality {model.bio_change_orth_coef:g}" if model.bio_change_orth_coef > 0 else ""))

	return model

