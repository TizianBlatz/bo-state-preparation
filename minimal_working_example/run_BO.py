import numpy as np
np.random.seed(237)
import matplotlib.pyplot as plt
from skopt.plots import plot_convergence

import skopt_wrapper as optimization
import interface as interface
import ed_simulation as ed


# ---- Optimization Parameters ----
num_iterations = 10			# Total calls to cost function from initialization + optimization
num_random_starts = 3		# Number of random points to initialize optimization

acq_func = 'LCB'			# acquisition function - 'LCB': lower confidence bound, 'EI': expected improvement
length_scale_bounds = (0.01, 100) # default: (0.01, 100)
num_exploration_sweeps = 1	# number of sweeps between exploration and exploitation
sweep_amplitude = 1			# amplitude of sweeps between exploration and exploitation

gp_noise = 1e-10			# manually tell optimizer about expected cost function noise level (do not set < 1e-10)

# ---- Simulation Parameters ----
T = 100						# total ramp time
num_linear_segments = 2		# number of linear ramp segments
save_dir = "data/"			# directory storing initial/target states and hamiltonians
t_shift = 60.0				# delay before turning on tunneling in the x-direction

# ---- Run BO ----
fqh_interface = interface.FQH_Plaquette(T=T, num_linear_segments=num_linear_segments, t_shift=t_shift, save_dir=save_dir) # interface class translating (4 * num_linear_segments)-dimensional landscape to ramps
optimized, true_optimizer = optimization.optimize(fqh_interface.cost, fqh_interface.param_bounds, num_linear_segments=fqh_interface.num_linear_segments, n_sweeps=num_exploration_sweeps, sweep_amplitude=sweep_amplitude, n_calls=num_iterations, n_random_starts=num_random_starts, acq_func=acq_func, save_dir=save_dir)

optimized_ramp = interface.set_segmented_ramp(optimized.x, T=T, num_linear_segments=num_linear_segments, t_shift=t_shift)

# ---- Plot Result ----
fig = plt.figure(figsize=(6.8, 3.5))
gs = fig.add_gridspec(1, 2)
ax_left = gs[0].subgridspec(1, 1).subplots()
axs_right = gs[1].subgridspec(2, 1).subplots()

plot_convergence(optimized, ax=ax_left)
optimized_ramp.plot(axs=axs_right)

ax_left.set_xlabel("Iteration")
ax_left.set_ylabel(r"Cost $1 - F^{1/2}$")
axs_right[0].set_title("Optimized ramp")

fig.tight_layout()
plt.show()

