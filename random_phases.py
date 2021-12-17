import itertools as it

from joblib import cpu_count, Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, special, integrate, optimize
import hankel

from phases import rvs_channel_phases, rvs_ris_phases, gains_constant_phase, rvs_ris_phases_quant
from utils import export_results


@np.vectorize
def inverse_exp_expi(y):
    def exp_expi(x):
        return -np.exp(x)*special.expi(-x)
    #sol = optimize.root_scalar(lambda x: (exp_expi(x)-y)**2, bracket=(0., np.inf))
    sol = optimize.minimize(lambda x: (exp_expi(x)-y)**2, (1e-16,), bounds=optimize.Bounds(0, np.inf))
    #print(sol)
    return sol.x


def ergodic_capac_approximation(num_elements, los_amp):
    if num_elements == 0:
        return np.log2(1 + los_amp**2)
    def _int_func(s, n, a):
        return np.log2(1.+s)*np.exp(-(a**2+s)/n)*special.i0(2*a/n*np.sqrt(s))
    _int = integrate.quad(_int_func, 0, np.inf, args=(num_elements, los_amp))
    erg_capac = _int[0]/num_elements
    return erg_capac

def ergodic_capac_exact(num_elements, los_amp=0.):
    if num_elements == 0:
        return 0.
    elif num_elements == 1:
        return 1. #log2(1+1)
    if los_amp > 0:
        raise NotImplementedError("Right now, only the NLOS case is supported.")
    nu = 0
    _int_func = lambda x: special.j0(x)**num_elements
    _h, _, _N = hankel.get_h(_int_func, nu=nu)
    ht = hankel.HankelTransform(nu=nu, N=_N, h=_h)
    if num_elements < 7:
        s = np.logspace(-7, np.log10(num_elements), 3000)
        Fs, err_hank = ht.transform(_int_func, k=s, ret_err=True)
        integrand = np.log2(1+s**2)*s*Fs
        cap_erg = integrate.simps(integrand, x=s)
    else:
        _quad_integrand = lambda x: x*np.log2(1+x**2)*ht.transform(_int_func, k=x, ret_err=False)
        _quad_int = integrate.quad(_quad_integrand, 0, num_elements, limit=2000, full_output=0)
        cap_erg = _quad_int[0]
    return cap_erg


def _process_batch(batch, num_batches, batch_size, num_elements, conn_prob,
                   los_amp, num_samples_fast, quant=None):
    print("Work on batch {:d}/{:d}".format(batch+1, num_batches))
    if los_amp > 0:
        los_phases = 2*np.pi*np.random.rand(batch_size)
        los_phases = np.tile(los_phases, (num_samples_fast, 1))
    else:
        los_phases = None
    channel_realizations = rvs_channel_phases(num_elements, batch_size)
    channel_realizations = np.tile(channel_realizations, (num_samples_fast, 1, 1))
    if quant is None or quant == 0:
        ris_phases = rvs_ris_phases(num_elements, batch_size,
                                    num_samples_fast, copula="indep")
    else:
        if not isinstance(quant, int): raise TypeError
        ris_phases = rvs_ris_phases_quant(num_elements, batch_size,
                                          num_samples_fast, copula="indep",
                                          K=quant)
    total_phases = channel_realizations + ris_phases
    channel_absolute = stats.bernoulli.rvs(p=conn_prob, size=(batch_size, num_elements))
    channel_absolute = np.tile(channel_absolute, (num_samples_fast, 1, 1))
    const_phase = gains_constant_phase(total_phases,
                                       los_phase=los_phases,
                                       los_amp=los_amp,
                                       path_amp=channel_absolute)
    capac_const_phase = np.log2(1 + const_phase)
    #expect_capac = np.append(expect_capac, np.mean(capac_const_phase, axis=0))
    return np.mean(capac_const_phase, axis=0)

def random_ris_phases(num_elements, connect_prob=[1.], los_amp=1., num_samples_slow=1000, num_samples_fast=5000,
                      plot=False, export=False, batch_size=1000, logplot=False,
                      parallel=False):
    if plot or logplot:
        fig, axs = plt.subplots()
    erg_capac_mc = []
    erg_capac_appr = []
    erg_cap_exact = []

    num_batches, last_batch = np.divmod(num_samples_slow, batch_size)
    #for _num_elements in num_elements:
    for _num_elements, _conn_prob in it.product(num_elements, connect_prob):
        print("Work on N={:d}, p={:.3f}".format(_num_elements, _conn_prob))
        results = {}
        if parallel:
            num_cores = cpu_count()
            expect_capac = Parallel(n_jobs=num_cores)(
                    delayed(_process_batch)(_batch, num_batches, batch_size,
                        _num_elements, _conn_prob, los_amp, num_samples_fast)
                    for _batch in range(num_batches))
            expect_capac = np.ravel(expect_capac)
        else:
            expect_capac = []
            for _batch in range(num_batches):
                __expect_cap = _process_batch(_batch, num_batches, batch_size,
                                              _num_elements, _conn_prob,
                                              los_amp, num_samples_fast)
                expect_capac = np.append(expect_capac, __expect_cap)
        #print(len(expect_capac))
        _erg_cap_mc = np.mean(expect_capac)
        print("Simulated ergodic capacity: {:.3f}".format(_erg_cap_mc))
        _hist = np.histogram(expect_capac, bins=100)
        _r_ax = np.linspace(0, 5, 2000)
        if logplot:
            _r_ax = np.logspace(np.log10(min(expect_capac)), np.log10(max(expect_capac)), 2000)
        _probs_cap_exact = [stats.binom(n=_num_elements, p=_conn_prob).pmf(__n) for __n in range(_num_elements+1)]
        if los_amp == 0.:
            #_erg_cap_appr = -np.exp(1/_num_elements)*special.expi(-1/_num_elements)/np.log(2)
            _inv_exi = inverse_exp_expi(_r_ax*np.log(2))
            cdf_appr = stats.binom(n=_num_elements, p=_conn_prob).cdf(1/_inv_exi)
            print("Calculating the exact ergodic capacities...")
            _erg_cap_exact = [ergodic_capac_exact(__n, los_amp=0.) for __n in range(_num_elements+1)]
            print("Exact ergodic capacity: {}".format(_erg_cap_exact))
            print("Exact ergodic probabilities: {}".format(_probs_cap_exact))
        else:
            _erg_cap_appr = [ergodic_capac_approximation(_n, los_amp) for _n in range(_num_elements+1)]
            print(_erg_cap_appr)
            cdf_appr = np.sum([np.heaviside(_r_ax-__erg_cap, 0)*_p
                               for _p, __erg_cap in zip(_probs_cap_exact, _erg_cap_appr)],
                              axis=0)
            _erg_cap_exact = [0.]
            _probs_cap_exact = [1.]
        #erg_capac_appr.append(_erg_cap_appr)
        #erg_cap_exact.append(_erg_cap_exact)
        #erg_capac_mc.append(_erg_cap_mc) # technically all expect_capac should be equal
        #print("Approximated ergodic capacity: {:.3f}".format(_erg_cap_appr))
        cdf_hist = stats.rv_histogram(_hist).cdf(_r_ax)
        #cdf_appr = np.heaviside(_r_ax-_erg_cap_appr, 0)
        cdf_exact = np.sum([np.heaviside(_r_ax-__erg_cap, 0)*_p
                            for _p, __erg_cap in zip(_probs_cap_exact, _erg_cap_exact)],
                           axis=0)
        if plot:
            axs.plot(_r_ax, cdf_hist, label="ECDF -- N={:d}, p={:.2f}".format(_num_elements, _conn_prob))
            axs.plot(_r_ax, cdf_appr, '--', label="Appr -- N={:d}, p={:.2f}".format(_num_elements, _conn_prob))
            axs.plot(_r_ax, cdf_exact, '-.', label="Exact -- N={:d}, p={:.2f}".format(_num_elements, _conn_prob))
            #axs.plot(_r_ax, stats.binom(n=_num_elements, p=_conn_prob).cdf(np.sqrt(2**_r_ax-1)), label="Optimal Phases")
        elif logplot:
            axs.semilogy(_r_ax, cdf_hist, label="ECDF -- N={:d}".format(_num_elements))
            axs.semilogy(_r_ax, cdf_appr, '--', label="Appr -- {:d}".format(_num_elements))
            axs.semilogy(_r_ax, cdf_exact, '-.', label="Exact -- {:d}".format(_num_elements))
        if export:
            results["rate"] = _r_ax
            results["ecdf"] = cdf_hist
            results["approx"] = cdf_appr
            results["exact"] = cdf_exact
            _fn_prefix = "out-prob-random-phase"
            _fn_mid = "los-a{:.2f}".format(los_amp) if los_amp > 0 else "nlos"
            _fn_end = "N{:d}-p{:.3f}".format(_num_elements, _conn_prob)
            _fn = "{}-{}-{}".format(_fn_prefix, _fn_mid, _fn_end)
            export_results(results, _fn)

    if plot or logplot:
        axs.legend()
        #axs.set_title("Artificial Fast Fading with N={:d} RIS Elements".format(num_elements))
        axs.set_xlabel("Rate $R$")
        axs.set_ylabel("Outage Probability $\\varepsilon$")

    #if export:
    #    erg_capac_results = {"N": num_elements, "mc": erg_capac_mc, "appr": erg_capac_appr}
    #    _fn_prefix = "erg-capac-random-phase"
    #    _fn_mid = "los-a{:.2f}".format(los_amp) if los_amp > 0 else "nlos"
    #    _fn_end = "N{:d}".format(_num_elements)
    #    _fn = "{}-{}-{}".format(_fn_prefix, _fn_mid, _fn_end)
    #    export_results(erg_capac_results, _fn)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--logplot", action="store_true")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("-N", "--num_elements", type=int, nargs="+", required=True)
    parser.add_argument("-f", "--num_samples_fast", type=int, default=5000)
    parser.add_argument("-s", "--num_samples_slow", type=int, default=1000)
    parser.add_argument("-b", "--batch_size", type=int, default=1000)
    parser.add_argument("-a", "--los_amp", type=float, default=0.)
    parser.add_argument("-p", "--connect_prob", type=float, nargs="+", default=[1.])
    args = vars(parser.parse_args())
    random_ris_phases(**args)
    plt.show()
