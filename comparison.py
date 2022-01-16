import itertools as it

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from random_phases import inverse_exp_expi
from constant_phases import outage_static_phases_approx, outage_static_phases_exact
from utils import export_results

def outage_prob_perfect_phases(rate, num_elements, conn_prob):
    return stats.binom(n=num_elements, p=conn_prob).cdf(np.sqrt(2**rate - 1))

def main(num_elements, connect_prob=[1.], plot=False, export=False):
    if plot:
        fig, axs = plt.subplots()
    rate = np.linspace(0, 5, 1000)
    _inv_exi = inverse_exp_expi(rate*np.log(2))
    for _num_elements, _conn_prob in it.product(num_elements, connect_prob):
        print("Work on N={:d}, p={:.3f}".format(_num_elements, _conn_prob))
        results = {}
        out_perfect = outage_prob_perfect_phases(rate, _num_elements, _conn_prob)
        out_static = outage_static_phases_approx(rate, _num_elements, _conn_prob)
        out_static_exact = outage_static_phases_exact(rate, _num_elements, _conn_prob)
        out_phase_hopping = stats.binom(n=_num_elements, p=_conn_prob).cdf(1/_inv_exi)

        if plot:
            axs.semilogy(rate, out_phase_hopping, label="Hopping -- N={:d}, p={:.3f}".format(_num_elements, _conn_prob))
            axs.semilogy(rate, out_perfect, '--', label="Perfect -- N={:d}, p={:.3f}".format(_num_elements, _conn_prob))
            axs.semilogy(rate, out_static, '-.', label="Static -- N={:d}, p={:.3f}".format(_num_elements, _conn_prob))
            axs.semilogy(rate, out_static_exact, '-.', label="Static (Exact) -- N={:d}, p={:.3f}".format(_num_elements, _conn_prob))
        if export:
            results["rate"] = rate
            results["hopping"] = out_phase_hopping
            results["perfect"] = out_perfect
            results["static"] = out_static_exact
            _fn_prefix = "out-prob-comparison"
            _fn_end = "N{:d}-p{:.3f}".format(_num_elements, _conn_prob)
            _fn = "{}-{}.dat".format(_fn_prefix, _fn_end)
            export_results(results, _fn)

    if plot:
        axs.legend()
        axs.set_xlabel("Transmission Rate $R$")
        axs.set_ylabel("Outage Probability $\\varepsilon$")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("-N", "--num_elements", type=int, nargs="+", required=True)
    parser.add_argument("-p", "--connect_prob", type=float, nargs="+", default=[1.])
    args = vars(parser.parse_args())
    main(**args)
    plt.show()
