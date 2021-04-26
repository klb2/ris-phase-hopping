import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special

from phases import gains_constant_phase, rvs_channel_phases, rvs_ris_phases_quant

def discrete_rvs_phases(num_elements, num_phase_steps, num_samples_slow=1000,
                        num_samples_fast=5000, plot=False):
    if plot:
        fig, axs = plt.subplots()
    dependency = "indep"
    channel_realizations = rvs_channel_phases(num_elements, num_samples_slow)
    channel_realizations = np.tile(channel_realizations, (num_samples_fast, 1, 1))
    ris_phases = rvs_ris_phases_quant(num_elements, num_samples_slow,
                                      num_samples_fast, copula=dependency,
                                      K=num_phase_steps)
    total_phases = channel_realizations + ris_phases
    const_phase = gains_constant_phase(total_phases)
    capac_const_phase = np.log2(1 + const_phase)
    expect_capac = np.mean(capac_const_phase, axis=0)
    _hist = np.histogram(expect_capac, bins=100)
    _r_ax = np.linspace(0, 3, 1000)
    cdf_hist = stats.rv_histogram(_hist).cdf(_r_ax)
    #cdf_exact = cdf_ergodic_capac(_r_ax, copula=dependency)
    if plot:
        axs.plot(_r_ax, cdf_hist, label="ECDF -- K={:d}".format(num_phase_steps))
        #axs.plot(_r_ax, cdf_exact, '--', label="Exact -- {}".format(dependency))
        axs.legend()
        axs.set_title("Artificial Fast Fading with N={:d} RIS Elements\nQuantized Phases with {:d} Steps".format(num_elements, num_phase_steps))
        axs.set_xlabel("Rate $R$")
        axs.set_ylabel("Outage Probability")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("-N", "--num_elements", type=int, default=2)
    parser.add_argument("-K", "--num_phase_steps", type=int, default=2)
    parser.add_argument("-f", "--num_samples_fast", type=int, default=5000)
    parser.add_argument("-s", "--num_samples_slow", type=int, default=1000)
    args = vars(parser.parse_args())
    discrete_rvs_phases(**args)
    plt.show()
