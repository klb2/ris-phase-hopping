import itertools as it

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from random_phases import ergodic_capac_approximation, ergodic_capac_exact
from utils import export_results


def main(num_elements, connect_prob, plot=False, export=False, **kwargs):
    eps = np.logspace(-9, -1, 200)
    for _num_el, _conn_prob in it.product(num_elements, connect_prob):
        results = {"eps": eps}
        print("Work on N={:d}, p={:.3f}".format(_num_el, _conn_prob))
        binom = stats.binom(n=_num_el, p=_conn_prob)
        print("Minimal outage probability: {:.3E}".format(binom.cdf(0)))
        _active_links = binom.ppf(eps)
        eps_cap_appr = [ergodic_capac_approximation(_n, los_amp=0.) for _n in _active_links]
        eps_cap_exact = [ergodic_capac_exact(_n, los_amp=0.) for _n in _active_links]
        eps_cap_optim = np.log2(1 + _active_links**2)
        results["exact"] = eps_cap_exact
        results["approx"] = eps_cap_appr
        results["optimal"] = eps_cap_optim
        if export:
            export_results(results, "eps-capac-N{:d}-p{:.3f}.dat".format(_num_el, _conn_prob))
        if plot:
            plt.semilogx(eps, eps_cap_exact, label="N={:d}, p={:.3f} (Exact)".format(_num_el, _conn_prob))
            plt.semilogx(eps, eps_cap_appr, '--', label="N={:d}, p={:.3f} (Approx)".format(_num_el, _conn_prob))
            plt.semilogx(eps, eps_cap_optim, '.-', label="N={:d}, p={:.3f} (Optimal)".format(_num_el, _conn_prob))
    if plot:
        plt.legend()
        plt.xlabel("Tolerated Outage Probability $\\varepsilon$")
        plt.ylabel("$\\varepsilon$-Outage Capacity $R^{\\varepsilon}$")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("-N", "--num_elements", type=int, nargs="+", required=True)
    parser.add_argument("-a", "--los_amp", type=float, default=0.)
    parser.add_argument("-p", "--connect_prob", type=float, nargs="+", default=[1.])
    args = vars(parser.parse_args())
    main(**args)
    plt.show()
