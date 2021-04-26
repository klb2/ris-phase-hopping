import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special

from phases import rvs_channel_phases, rvs_ris_phases, gains_constant_phase

def cdf_ergodic_capac_two_elements(rate, copula="indep"):
    if copula.startswith("comon"):
        #pdf = 2./(np.pi*np.sqrt(2-2**rate))
        cdf = 2/np.pi * np.arcsin(0.5 * np.sqrt(2**rate-1))
        cdf[np.isnan(cdf)] = 1.
    elif copula.startswith("indep"):
        #ergodic = -(np.exp(1/2)*special.expi(-1/2))/(np.log(2)) #approximation
        ergodic = np.arccosh(3/2)/np.log(2)
        cdf = np.heaviside(rate-ergodic, .5)
    elif copula.startswith("counter"):
        ergodic = np.arccosh(3/2)/np.log(2)
        cdf = np.heaviside(rate-ergodic, .5)
    return cdf

def random_ris_phases(num_elements, num_samples_slow=1000, num_samples_fast=5000, plot=True):
    if plot:
        fig, axs = plt.subplots()
    dependencies = ["comon", "indep"]
    if num_elements == 2:
        dependencies.append("counter")
    channel_realizations = rvs_channel_phases(num_elements, num_samples_slow)
    channel_realizations = np.tile(channel_realizations, (num_samples_fast, 1, 1))
    for _dependency in dependencies:
        ris_phases = rvs_ris_phases(num_elements, num_samples_slow,
                                    num_samples_fast, copula=_dependency)
        print("Dependency: {}".format(_dependency))
        total_phases = channel_realizations + ris_phases
        const_phase = gains_constant_phase(total_phases)
        capac_const_phase = np.log2(1 + const_phase)
        expect_capac = np.mean(capac_const_phase, axis=0)
        _hist = np.histogram(expect_capac, bins=100)
        _r_ax = np.linspace(0, 3, 1000)
        cdf_hist = stats.rv_histogram(_hist).cdf(_r_ax)
        if num_elements == 2:
            cdf_exact = cdf_ergodic_capac_two_elements(_r_ax, copula=_dependency)
        if plot:
            axs.plot(_r_ax, cdf_hist, label="ECDF -- {}".format(_dependency))
            if num_elements == 2:
                axs.plot(_r_ax, cdf_exact, '--', label="Exact -- {}".format(_dependency))
    if plot:
        axs.legend()
        axs.set_title("Artificial Fast Fading with N={:d} RIS Elements".format(num_elements))
        axs.set_xlabel("Rate $R$")
        axs.set_ylabel("Outage Probability")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("-N", "--num_elements", type=int, default=2)
    parser.add_argument("-f", "--num_samples_fast", type=int, default=5000)
    parser.add_argument("-s", "--num_samples_slow", type=int, default=1000)
    args = vars(parser.parse_args())
    random_ris_phases(**args)
    plt.show()
