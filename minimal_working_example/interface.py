import numpy as np
import matplotlib.pyplot as plt
import bisect
import collections

import ed_simulation as ed


class Interface():
	def __init__(self, params, cost):
		self.params = params

	def cost(self):
		raise Exception("Cost function not implemented.")


class FQH_Plaquette(Interface):
	def __init__(self, T = np.nan, num_linear_segments = -1, t_shift = np.nan, save_dir = ""):
		self.num_linear_segments = num_linear_segments
		self.T = T
		self.t_shift = t_shift

		self.num_ramp_parameters = 4
		self.segment_params = collections.OrderedDict({
			"tx" : (0.0, 1.2), "ty" : (0.0, 1.2), "delta_x" : (-4.0, 4.0), "delta_y" : (-4.0, 4.0),
		})

		self.param_bounds = list(self.segment_params.values()) * (self.num_linear_segments - 1)
		self.save_dir = save_dir

		return

	def cost(self, param_list):
		ramp = self.param_list_to_ramp(param_list)

		fidelity = self.ramp_fidelity(ramp, verbose=True)
		return 1 - np.sqrt(fidelity)

	def ramp_fidelity(self, ramp, verbose = False):
		return ed.fidelity(ramp, verbose=verbose, save_dir=self.save_dir)

	def param_list_to_ramp(self, param_list):
		return set_segmented_ramp(param_list.copy(), T=self.T, num_linear_segments=self.num_linear_segments, num_ramp_parameters=self.num_ramp_parameters, t_shift=self.t_shift)

	def ramp_to_param_list(self, ramp):
		num_params_total = self.num_ramp_parameters * (self.num_linear_segments - 1)
		param_list = [None for i in range(num_params_total)]

		for i in range(self.num_linear_segments - 1):
			param_list[i * self.num_ramp_parameters] = ramp.t_xs[i+1]
			param_list[i * self.num_ramp_parameters + 1] = ramp.t_ys[i+1]
			param_list[i * self.num_ramp_parameters + 2] = ramp.delta_xs[i+1]
			param_list[i * self.num_ramp_parameters + 3] = ramp.delta_ys[i+1]

		return param_list


class Ramp(object):
	def __init__(self, times = [], t_xs = [], t_ys = [], delta_xs = [], delta_ys = [], eval_times = [], t_shift = np.nan):
		param_lists = [t_xs, t_ys, delta_xs, delta_ys]
		sorted_param_lists = []
		for param_list in param_lists:
			if param_list and len(param_list) != len(times):
				raise Exception("All parameter lists must have the same dimension as the time list")
			sorted_param_lists.append([x for _, x in sorted(zip(times, param_list))])

		[self.t_xs, self.t_ys, self.delta_xs, self.delta_ys] = sorted_param_lists
		self.times = sorted(times)
		self.eval_times = sorted(set(eval_times)) if eval_times else list(sorted(set(self.times)))
		self.t_shift = t_shift
		self.fidelity = None

	def __linear_interpolation__(self, time, params):
		i = bisect.bisect(self.times, time)

		if i == len(self.times) or abs(time - params[i-1]) < 1e-8:
			return params[i-1]
		elif i == 0 or abs(time - params[i]) < 1e-8:
			return params[i]

		# do linear interpolation
		t_below, t_above = self.times[i-1], self.times[i]
		p_below, p_above = params[i-1], params[i]

		return p_below + (p_above - p_below) * (time - t_below) / (t_above - t_below)

	def __shift_time__(self, time, inverse = False):
		if inverse:
			t_eff = self.t_shift + time * (self.times[-1] - self.t_shift) / self.times[-1]
			return t_eff

		if time <= self.t_shift:
			return 0.0

		t_eff = (time - self.t_shift) / (self.times[-1] - self.t_shift) * self.times[-1]
		return t_eff

	def t_x(self, time):
		return self.__linear_interpolation__(self.__shift_time__(time), self.t_xs)

	def t_y(self, time):
		return self.__linear_interpolation__(time, self.t_ys)

	def delta_x(self, time):
		return self.__linear_interpolation__(self.__shift_time__(time), self.delta_xs)

	def delta_y(self, time):
		return self.__linear_interpolation__(time, self.delta_ys)

	def print(self, file = None):
		print("	times = 	", [f"{i:.2f}" for i in self.times], file=file)
		print("	t_xs = 		", [f"{i:.2f}" for i in self.t_xs], file=file)
		print("	t_ys = 		", [f"{i:.2f}" for i in self.t_ys], file=file)
		print("	delta_xs =	", [f"{i:.2f}" for i in self.delta_xs], file=file)
		print("	delta_ys = 	", [f"{i:.2f}" for i in self.delta_ys], file=file)
		return

	def plot(self, axs = [], ramp_type="optimal", colors = ["#0F569A", "#C44006", "#4EA6DA", "#F47313"]):
		new_fig = len(axs) == 0
		if new_fig:
			fig, axs = plt.subplots(2, 1, sharex="col", figsize=(5, 4))

		dense_times = np.linspace(self.times[0], self.times[-1], 1000)
		l1, = axs[0].plot(dense_times, [self.t_x(t) for t in dense_times], label=r"$t_x$", lw=2, ms=0, marker="o", mfc="white", color=colors[0], ls="solid")
		l2, = axs[0].plot(dense_times, [self.t_y(t) for t in dense_times], label=r"$t_y$", lw=2, ms=0, marker="o", mfc="white", color=colors[1], ls="solid")
		l3, = axs[1].plot(dense_times, [self.delta_x(t) for t in dense_times], label=r"$\Delta_x$", lw=2, ms=0, marker="o", mfc="white", color=colors[2], ls="solid")
		l4, = axs[1].plot(dense_times, [self.delta_y(t) for t in dense_times], label=r"$\Delta_y$", lw=2, ms=0, marker="o", mfc="white", color=colors[3], ls="solid")

		controlled_time_points = self.times[1:-1]
		shifted_time_points = [self.__shift_time__(t, inverse=True) for t in controlled_time_points]
		if not ramp_type == "experimental":
			axs[0].plot(shifted_time_points , [self.t_x(t) for t in shifted_time_points], label=None, lw=0, marker="o", mfc="white", color=colors[0], ls="solid")
			axs[0].plot(controlled_time_points, [self.t_y(t) for t in controlled_time_points], label=None, lw=0, marker="o", mfc="white", color=colors[1], ls="solid")
			axs[1].plot(shifted_time_points, [self.delta_x(t) for t in shifted_time_points], label=None, lw=0, marker="o", mfc="white", color=colors[2], ls="solid")
			axs[1].plot(controlled_time_points, [self.delta_y(t) for t in controlled_time_points], label=None, lw=0, marker="o", mfc="white", color=colors[3], ls="solid")

		for ax in axs:
			for time in self.times + shifted_time_points:
				ax.axvline(time, color="grey", lw=0.5, ls="dashed", zorder=-1)
			ax.axvline(self.t_shift, color="grey", lw=0.5, zorder=-1)
		for ax in axs:
			ax.axhline(0.0, color="grey", lw=0.5, ls="solid", zorder=-1)

		axs[0].set_ylim([-0.05, 1.1 * max(self.t_xs + self.t_ys)])
		axs[-1].set_xlabel(r"Time  $[\tau]$")
		axs[0].set_ylabel("Tunneling\n" + r"$t$  [$\hbar / \tau$]")
		axs[1].set_ylabel("Tilt per site\n" + r"$\Delta$  [$\hbar / \tau$]")

		axs[0].legend(loc="lower right")
		axs[1].legend(loc="upper right")

		for ax in axs:
			if new_fig:
				ax.set_xlim([self.times[0], self.times[-1]])
			ax.tick_params(which="both", direction="in", top="on", right="on")

		if new_fig:
			fig.tight_layout()
			plt.show()
		return axs


# set a ramp based on a parameter list containing all parameters. Assume the ordering of parameters
# is as in the interface class.
#
# param list – ramp parameters flattened in list
# T – total ramp time
# num_linear_segments – number of linear segments of the ramps -> N-1 optimized points between fixed start and end
def set_segmented_ramp(param_list, T = np.nan, num_linear_segments = -1, num_ramp_parameters = -1, t_shift = np.nan):
	# fix initial parameters
	times =		[0.0]
	t_xs = 		[0.0]
	t_ys =		[0.0]
	delta_xs = [2.0]
	delta_ys = [1.0] 

	for i in range(num_linear_segments - 1):
		times.append((i+1) * T / num_linear_segments)
		t_xs.append(param_list[i * num_ramp_parameters])
		t_ys.append(param_list[i * num_ramp_parameters + 1])
		delta_xs.append(param_list[i * num_ramp_parameters + 2])
		delta_ys.append(param_list[i * num_ramp_parameters + 3])

	# fix target parameters
	times.append(T)
	t_xs.append(1)
	t_ys.append(1)
	delta_xs.append(0.0)
	delta_ys.append(0.0)

	return Ramp(times, t_xs, t_ys, delta_xs, delta_ys, t_shift=t_shift)


def set_experimental_ramp(T = 100.0):
	times =		[0, 20, 60, 70,  90,  100]
	t_xs = 		[0, 0,  0,  1.2, 1.2, 1]
	t_ys =		[0, 1,  1,  1,   1,   1]
	delta_xs = 	[1, 1,  1,  1,   0,   0]
	delta_ys =	[4, 4,  0,  0,   0,   0]

	eval_times = [0, 40, 60, 75, 90, 95, 100]

	return Ramp(times, t_xs, t_ys, delta_xs, delta_ys, eval_times)

