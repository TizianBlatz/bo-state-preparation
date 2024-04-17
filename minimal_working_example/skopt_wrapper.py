####################################################################################################
# Wrapper of skopt functionalities exposing the length-scale bounds of the GP model and allowing
# the iteration-by-iteration control needed for sweeping the hyperparameter kappa.
####################################################################################################

import skopt
import skopt.learning.gaussian_process.kernels as kernels
import numpy as np
import pickle as pkl
import copy
import numbers
import inspect
import warnings
from collections.abc import Iterable
from pathlib import Path


"""
BO optimization loop - adapted from the gp_minimize functionality, with added iteration-by-iteration
control and exposing length-scale bounds
"""
def optimize(cost, param_bounds = None, num_linear_segments = np.nan,
	length_scale_bounds = (0.01, 100), n_sweeps = -1, sweep_amplitude = 1, n_calls = -1, n_random_starts = 20, x0 = None,
	noise = "gaussian", acq_func = "gp_hedge", random_state = 1111, fix_start_params = True, optimize_global_params = False,
	verbose = True, save_dir = ""):

	if verbose:
		print("Bounds on parameters: ", param_bounds)

# ---- Initialize Opimization ---- #
	optimizer, res, specs, callbacks = init_optimization(cost, param_bounds, length_scale_bounds=length_scale_bounds,
		n_calls=n_calls, n_random_starts=n_random_starts, x0=x0, noise=noise, verbose=verbose, acq_func=acq_func, random_state=random_state)

# ---- Optimize ---- #
	default_xi = 0.01
	default_kappa = 1.96
	period = 2 / (2*n_sweeps - 1)
	for n in range(n_calls):
		if n_sweeps > 0: # sweep between exploration and exploitation
			exponent = sweep_amplitude * triangle_wave(n/n_calls, period=period)
			xi = default_xi * 10 ** exponent
			kappa = default_kappa * 10 ** exponent
			optimizer.acq_func_kwargs = {"xi" : xi, "kappa" : kappa}
		next_x = optimizer.ask()
		next_y = cost(next_x)
		res = optimizer.tell(next_x, next_y)
		res.specs = specs

		if skopt.utils.eval_callbacks(callbacks, res):
			break

	return res, optimizer


"""
Initializ optimization similar to the base_minimize method in skopt.optimizer.base - expose length scale bounds
"""
def init_optimization(cost=None, dimensions=None, base_estimator=None, length_scale_bounds = (0.01, 100),
				n_calls = -1, n_random_starts=None, n_initial_points=10, ask = False,
				initial_point_generator="random",
				acq_func="gp_hedge", acq_optimizer="auto", x0=None, y0=None,
				random_state=None, verbose=False, callback=None,
				n_points=10000, n_restarts_optimizer=5, xi=0.01, kappa=1.96,
				noise="gaussian", n_jobs=1, model_queue_size=None, models=[], save_dir = ""):

	# Check params
	rng = skopt.utils.check_random_state(random_state)
	space = skopt.utils.normalize_dimensions(dimensions)

	specs = {"args": copy.copy(inspect.currentframe().f_locals),
			"function": inspect.currentframe().f_code.co_name}

	base_estimator = estimator(
		"GP", length_scale_bounds=length_scale_bounds, space=space, random_state=rng.randint(0, np.iinfo(np.int32).max),
		noise=noise)

	acq_optimizer_kwargs = {
		"n_points": n_points, "n_restarts_optimizer": n_restarts_optimizer,
		"n_jobs": n_jobs}
	acq_func_kwargs = {"xi": xi, "kappa": kappa}

# ---- Initialize optimization ---- #
	# Suppose there are points provided (x0 and y0), record them
	# check x0: list-like, requirement of minimal points
	if x0 is None:
		x0 = []
	elif not isinstance(x0[0], (list, tuple)):
		x0 = [x0]
	if not isinstance(x0, list):
		raise ValueError("`x0` should be a list, but got %s" % type(x0))

	# Check `n_random_starts` deprecation first
	if n_random_starts is not None:
		warnings.warn(("n_random_starts will be removed in favour of "
				"n_initial_points. It overwrites n_initial_points."),
				DeprecationWarning)
		n_initial_points = n_random_starts

	if n_initial_points <= 0 and not x0:
		raise ValueError("Either set `n_initial_points` > 0,"
						" or provide `x0`")
	# check y0: list-like, requirement of maximal calls
	if isinstance(y0, Iterable):
		y0 = list(y0)
	elif isinstance(y0, numbers.Number):
		y0 = [y0]

	# calculate the total number of initial points
	n_initial_points = n_initial_points + len(x0)

# ---- Build optimizer ---- #
	optimizer = skopt.Optimizer(dimensions, base_estimator,
								n_initial_points=n_initial_points,
								initial_point_generator=initial_point_generator,
								n_jobs=n_jobs,
								acq_func=acq_func, acq_optimizer=acq_optimizer,
								random_state=random_state,
								model_queue_size=model_queue_size,
								acq_optimizer_kwargs=acq_optimizer_kwargs,
								acq_func_kwargs=acq_func_kwargs)

	# check x0: element-wise data type, dimensionality
	assert all(isinstance(p, Iterable) for p in x0)
	if not all(len(p) == optimizer.space.n_dims for p in x0):
		raise RuntimeError("Optimization space (%s) and initial points in x0 "
				"use inconsistent dimensions." % optimizer.space)

	# check callback
	callbacks = skopt.callbacks.check_callback(callback)
	if verbose:
		callbacks.append(skopt.callbacks.VerboseCallback(
			n_init=len(x0) if not y0 else 0,
			n_random=n_initial_points,
			n_total=n_calls))

# ---- Record provided points ---- #
	# create return object
	result = None
	# evaluate y0 if only x0 is provided
	if x0 and y0 is None:
		y0 = list(map(cost, x0))
		n_calls -= len(y0)
	# record through tell function
	if x0:
		if not (isinstance(y0, Iterable) or isinstance(y0, numbers.Number)):
			raise ValueError("`y0` should be an iterable or a scalar, got %s" % type(y0))
		if len(x0) != len(y0):
			raise ValueError("`x0` and `y0` should have the same length")
		result = optimizer.tell(x0, y0)
		result.specs = specs

	if models:
		optimizer.models = models
		result.models = models
	if not ask:
		return optimizer, result, specs, callbacks

	next_x = optimizer.ask()
	if save_dir:
		skopt.dump(result, save_dir + "optimized.opt")
		pkl.dump(optimizer, open(save_dir + "optimizer.opt", "wb"))
		pkl.dump(specs, open(save_dir + "specs.opt", "wb"))
		pkl.dump(callbacks, open(save_dir + "callbacks.opt", "wb"))
	return next_x


"""
Build extimator - derivative of the cook_estimator method in skopt.utils - expose length scale bounds
"""
def estimator(base_estimator, length_scale_bounds = (0.01, 100), space=None, **kwargs):
	"""Cook a default estimator.

	For the special base_estimator called "DUMMY" the return value is None.
	This corresponds to sampling points at random, hence there is no need
	for an estimator.

	Parameters
	----------
	base_estimator : "GP", "RF", "ET", "GBRT", "DUMMY" or sklearn regressor
		Should inherit from `sklearn.base.RegressorMixin`.
		In addition the `predict` method should have an optional `return_std`
		argument, which returns `std(Y | x)`` along with `E[Y | x]`.
		If base_estimator is one of ["GP", "RF", "ET", "GBRT", "DUMMY"], a
		surrogate model corresponding to the relevant `X_minimize` function
		is created.

	space : Space instance
		Has to be provided if the base_estimator is a gaussian process.
		Ignored otherwise.

	kwargs : dict
		Extra parameters provided to the base_estimator at init time.
	"""
	if isinstance(base_estimator, str):
		base_estimator = base_estimator.upper()
		if base_estimator not in ["GP", "ET", "RF", "GBRT", "DUMMY"]:
			raise ValueError("Valid strings for the base_estimator parameter "
							" are: 'RF', 'ET', 'GP', 'GBRT' or 'DUMMY' not "
							"%s." % base_estimator)
	elif not is_regressor(base_estimator):
		raise ValueError("base_estimator has to be a regressor.")

	if base_estimator == "GP":
		if space is not None:
			space = skopt.space.Space(space)
			space = skopt.space.Space(skopt.utils.normalize_dimensions(space.dimensions))
			n_dims = space.transformed_n_dims
			is_cat = space.is_categorical
		else:
			raise ValueError("Expected a Space instance, not None.")

		cov_amplitude = kernels.ConstantKernel(1.0, (0.01, 1000.0))
		# only special if *all* dimensions are categorical
		if is_cat:
			other_kernel = kernels.HammingKernel(length_scale=np.ones(n_dims))
		else:
			other_kernel = kernels.Matern(
			length_scale=np.ones(n_dims),
			length_scale_bounds=[length_scale_bounds] * n_dims, nu=2.5)

		base_estimator = skopt.learning.GaussianProcessRegressor(
			kernel=cov_amplitude * other_kernel,
			normalize_y=True, noise="gaussian",
			n_restarts_optimizer=2)
	elif base_estimator == "RF":
		base_estimator = skopt.learning.RandomForestRegressor(n_estimators=100, min_samples_leaf=3)
	elif base_estimator == "ET":
		base_estimator = skopt.learning.ExtraTreesRegressor(n_estimators=100, min_samples_leaf=3)
	elif base_estimator == "GBRT":
		gbrt = skopt.learning.GradientBoostingRegressor(n_estimators=30, loss="quantile")
		base_estimator = skopt.learning.GradientBoostingQuantileRegressor(base_estimator=gbrt)

	elif base_estimator == "DUMMY":
		return None

	if ('n_jobs' in kwargs.keys()) and not hasattr(base_estimator, 'n_jobs'):
		del kwargs['n_jobs']

	base_estimator.set_params(**kwargs)
	return base_estimator


def triangle_wave(x, period = np.nan): # goes between 1 and -1
	return 1 - 4*np.abs(x/period - np.floor(x/period + 0.5))


def save(optimized, filename = "optimized.opt", save_dir = ""):
	Path(save_dir).mkdir(parents=True, exist_ok=True)
	skopt.dump(optimized, save_dir + filename)


