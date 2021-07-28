import numpy as np
import matplotlib.pyplot as plt
from scipy import special

from random_phases import ergodic_capac_approximation, ergodic_capac_exact


def plot_ergodic_capacities_nlos_phase_hopping(num_elements, los_amp=0.,
                                               plot=False):
    erg_capac_appr = []
    erg_cap_exact = []
    for _num_elements in num_elements:
        print("N={:d}".format(_num_elements))
        if los_amp == 0.:
            _erg_cap_appr = -np.exp(1/_num_elements)*special.expi(-1/_num_elements)/np.log(2)
            _erg_cap_exact = ergodic_capac_exact(_num_elements, los_amp=0.)
            print("Exact ergodic capacity: {:.3f}".format(_erg_cap_exact))
        else:
            _erg_cap_appr = ergodic_capac_approximation(_num_elements, los_amp)
            _erg_cap_exact = 0.
        print("Approximate ergodic capacity: {:.3f}".format(_erg_cap_appr))
        erg_capac_appr.append(_erg_cap_appr)
        erg_cap_exact.append(_erg_cap_exact)
    if plot:
        fig, axs = plt.subplots()
        axs.plot(num_elements, erg_cap_exact, label="Exact")
        axs.plot(num_elements, erg_capac_appr, label="Approximation")
        axs.legend()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("-N", "--num_elements", type=int, nargs="+", default=[2, 3, 5, 10, 50])
    parser.add_argument("-a", "--los_amp", type=float, default=0.)
    args = vars(parser.parse_args())
    plot_ergodic_capacities_nlos_phase_hopping(**args)
    plt.show()
