import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from phases import gains_constant_phase, rvs_channel_phases, rvs_ris_phases

def constant_ris_phases(num_elements, num_samples=50000, plot=False):
    if plot:
        fig, axs = plt.subplots()
    for _num_elements in num_elements:
        channel_realizations = rvs_channel_phases(_num_elements, num_samples)
        const_phase = gains_constant_phase(channel_realizations)
        capac_const_phase = np.log2(1 + const_phase)
        _hist = np.histogram(capac_const_phase, bins=100)
        _r_ax = np.linspace(min(capac_const_phase)*.9, max(capac_const_phase)*1.1, 1000)
        cdf_hist = stats.rv_histogram(_hist).cdf(_r_ax)
        cdf_appr = 1. - np.exp(-(2**_r_ax-1)/_num_elements)  # N --> oo, for sum
        if plot:
            axs.plot(_r_ax, cdf_hist, label="Empirical CDF N={:d}".format(_num_elements))
            axs.plot(_r_ax, cdf_appr, '--', label="Approximate N={:d}".format(_num_elements))
    if plot:
        axs.set_xlabel("Rate $R$")
        axs.set_ylabel("Outage Probability")
        axs.legend()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("-N", "--num_elements", nargs="+", type=int, required=True)
    parser.add_argument("-n", "--num_samples", type=int, default=50000)
    args = vars(parser.parse_args())
    constant_ris_phases(**args)
    plt.show()