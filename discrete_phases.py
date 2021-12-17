from joblib import cpu_count, Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special

from phases import gains_constant_phase, rvs_channel_phases, rvs_ris_phases_quant
from random_phases import _process_batch
from utils import export_results

def discrete_rvs_phases(num_elements, num_phase_steps, connect_prob,
                        num_samples_slow=1000, num_samples_fast=5000,
                        batch_size=1000, plot=False, export=False,
                        parallel=False):
    if plot:
        fig, axs = plt.subplots()
    channel_realizations = rvs_channel_phases(num_elements, num_samples_slow)
    channel_realizations = np.tile(channel_realizations, (num_samples_fast, 1, 1))
    results = {}
    #_r_ax = np.linspace(0.5, 4, 2000)
    _r_ax = np.linspace(0, 3, 2000)

    num_batches, last_batch = np.divmod(num_samples_slow, batch_size)
    for _conn in connect_prob:
        print("Working on p={:.3f}".format(_conn))
        #channel_absolute = stats.bernoulli.rvs(p=_conn, size=(num_samples_slow, num_elements))
        #channel_absolute = np.tile(channel_absolute, (num_samples_fast, 1, 1))
        for _K in num_phase_steps:
            print("Working on K={:d}".format(_K))
            #ris_phases = rvs_ris_phases_quant(num_elements, num_samples_slow,
            #                                  num_samples_fast, copula="indep",
            #                                  K=_K)
            #total_phases = channel_realizations + ris_phases
            #const_phase = gains_constant_phase(total_phases, path_amp=channel_absolute)
            #capac_const_phase = np.log2(1 + const_phase)
            #expect_capac = np.mean(capac_const_phase, axis=0)
            if parallel:
                num_cores = cpu_count()
                expect_capac = Parallel(n_jobs=num_cores)(
                        delayed(_process_batch)(_batch, num_batches, batch_size,
                                                num_elements, _conn, 0., num_samples_fast)
                        for _batch in range(num_batches))
                expect_capac = np.ravel(expect_capac)
            else:
                expect_capac = []
                for _batch in range(num_batches):
                    __expect_cap= _process_batch(_batch, num_batches, batch_size,
                                                num_elements, _conn, 0.,
                                                num_samples_fast, quant=_K)
                    expect_capac = np.append(expect_capac, __expect_cap)

            print("K={:d}: ZOC={:.3f}".format(_K, min(expect_capac)))
            #_r_ax = np.linspace(.9*min(expect_capac), 1.1*max(expect_capac), 500)
            _hist = np.histogram(expect_capac, bins=100)
            cdf_hist = stats.rv_histogram(_hist).cdf(_r_ax)
            _erg_cap_appr = -np.exp(1/num_elements)*special.expi(-1/num_elements)/np.log(2)
            cdf_appr = np.heaviside(_r_ax-_erg_cap_appr, 0)
            results["ecdf{:d}".format(_K)] = cdf_hist
            results["appr{:d}".format(_K)] = cdf_appr
            if plot:
                axs.plot(_r_ax, cdf_hist, label="K={:d}, p={:.3f}".format(_K, _conn))

        if export:
            results["rate"] = _r_ax
            export_results(results, "quant-phase-N{:d}-p{:.3f}".format(num_elements, _conn))

    if plot:
        axs.legend()
        axs.set_title("Artificial Fast Fading with N={:d} RIS Elements\nQuantized Phases".format(num_elements))
        axs.set_xlabel("Rate $R$")
        axs.set_ylabel("Outage Probability")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("-N", "--num_elements", type=int, default=2)
    parser.add_argument("-K", "--num_phase_steps", type=int, nargs="+", default=[2, 3, 5, 10])
    parser.add_argument("-f", "--num_samples_fast", type=int, default=5000)
    parser.add_argument("-s", "--num_samples_slow", type=int, default=1000)
    parser.add_argument("-b", "--batch_size", type=int, default=1000)
    parser.add_argument("-p", "--connect_prob", type=float, nargs="+", default=[1.])
    args = vars(parser.parse_args())
    discrete_rvs_phases(**args)
    plt.show()
