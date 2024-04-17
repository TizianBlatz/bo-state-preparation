import time
import numpy as np
import pickle as pkl
from scipy import integrate


def fidelity(ramp, U = 8, alpha = 0.27, verbose = True, save_dir = ""):
	t_start = time.perf_counter()
	eval_times = ramp.eval_times.copy()

	if verbose:
		print(f"\n Calculate fidelity at ")
		ramp.print()
		print("simulating ramp...")

	# ---- Load initial- and target state as well as the matrices for the different Hamiltonian terms ----
	psi_initial = np.load(save_dir + "initial_state.npy")
	psi_target = np.load(save_dir + "target_state.npy")

	h_kin_x_mat = pkl.load(open(save_dir + f"hamiltonians/alpha{alpha:.3f}/h_kin_x.op", "rb"))
	h_kin_y = pkl.load(open(save_dir + f"hamiltonians/alpha{0.0:.3f}/h_kin_y.op", "rb"))
	h_int = pkl.load(open(save_dir + "hamiltonians/h_int.op", "rb"))
	h_delta_x = pkl.load(open(save_dir + "hamiltonians/h_delta_x.op", "rb"))
	h_delta_y = pkl.load(open(save_dir + "hamiltonians/h_delta_y.op", "rb"))

	# ---- Perform ED time evolution using the Runge-Kutta method ----
	t_beg = ramp.times[0]
	t_end = ramp.times[-1]
	print(f"evolving to T = {t_end}...")
	seq_lhs = lambda time, psi_t: -1j * harper_hofstadter_time_dep(time, ramp, U=U, h_kin_x_mat=h_kin_x_mat, h_kin_y=h_kin_y, h_int=h_int, h_delta_x=h_delta_x, h_delta_y=h_delta_y)@psi_t
	sol = integrate.solve_ivp(seq_lhs, (t_beg, t_end), psi_initial.copy(), t_eval=eval_times, max_step=1e-1)
	evolved_states = (sol.y).copy()

	fidelity = abs(overlap(psi_target, evolved_states[:, -1])) ** 2
	fidelity /= abs(overlap(psi_target, psi_target)) * abs(overlap(evolved_states[:, -1], evolved_states[:, -1]))

	if verbose and True:
		print(f"Simulation took {(time.perf_counter() - t_start):.1f}s")
		print(f"	F = ", fidelity, "\n")

	return fidelity


def harper_hofstadter(K = 0, J = 0, U = 8, delta_x = 0.0, delta_y = 0.0, h_kin_x_mat = None, h_kin_y = None, h_int = None, h_delta_x = None, h_delta_y = None):
	H = K * h_kin_x_mat + J * h_kin_y + U * h_int + delta_x * h_delta_x + delta_y * h_delta_y
	return H


def harper_hofstadter_time_dep(time, ramp, U = np.nan, h_kin_x_mat = None, h_kin_y = None, h_int = None, h_delta_x = None, h_delta_y = None, fast = True, save = True, save_dir = ""):
	return harper_hofstadter(K=ramp.t_x(time), J=ramp.t_y(time), U=U, delta_x=ramp.delta_x(time), delta_y=ramp.delta_y(time), h_kin_x_mat=h_kin_x_mat, h_kin_y=h_kin_y, h_int=h_int, h_delta_x=h_delta_x, h_delta_y=h_delta_y)


"""
Calculate the overlap <state1|state2> of states 1 and 2 assuming they are in the same basis
"""
def overlap(state1, state2):
	return np.conjugate(state1.transpose())@state2

