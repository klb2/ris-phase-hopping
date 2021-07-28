import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from phases import gains_constant_phase, rvs_channel_phases, rvs_ris_phases
from utils import export_results

def constant_ris_phases(num_elements, los_amp=1., num_samples=50000, plot=False,
                        export=False):
    if plot:
        fig, axs = plt.subplots()
    los_phases = 2*np.pi*np.random.rand(num_samples) if los_amp > 0 else None
    for _num_elements in num_elements:
        print("Work on N={:d}".format(_num_elements))
        results = {}
        channel_realizations = rvs_channel_phases(_num_elements, num_samples)
        const_phase = gains_constant_phase(channel_realizations, los_phase=los_phases, los_amp=los_amp)
        capac_const_phase = np.log2(1 + const_phase)
        _hist = np.histogram(capac_const_phase, bins=100)
        _r_ax = np.linspace(min(capac_const_phase)*.9, max(capac_const_phase)*1.1, 1000)
        cdf_hist = stats.rv_histogram(_hist).cdf(_r_ax)
        if los_amp == 0.:
            cdf_appr = 1. - np.exp(-(2**_r_ax-1)/_num_elements)  # N --> oo, for sum
        else:
            cdf_appr = stats.ncx2(df=2, nc=2*los_amp**2/_num_elements).cdf(2*(2**_r_ax-1)/_num_elements)
        if plot:
            axs.plot(_r_ax, cdf_hist, label="Empirical CDF N={:d}".format(_num_elements))
            axs.plot(_r_ax, cdf_appr, '--', label="Approximate N={:d}".format(_num_elements))
        if export:
            results["rate"] = _r_ax
            results["ecdf"] = cdf_hist
            results["approx"] = cdf_appr
            _fn_prefix = "out-prob-const-phase"
            _fn_mid = "los-a{:.2f}".format(los_amp) if los_amp > 0 else "nlos"
            _fn_end = "N{:d}".format(_num_elements)
            _fn = "{}-{}-{}".format(_fn_prefix, _fn_mid, _fn_end)
            export_results(results, _fn)
    if plot:
        axs.set_xlabel("Rate $R$")
        axs.set_ylabel("Outage Probability")
        axs.legend()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("-N", "--num_elements", nargs="+", type=int, required=True)
    parser.add_argument("-n", "--num_samples", type=int, default=50000)
    parser.add_argument("-a", "--los_amp", type=float, default=1.)
    args = vars(parser.parse_args())
    constant_ris_phases(**args)
    plt.show()
