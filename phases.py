import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from rearrangement_algorithm import basic_rearrange

def rvs_channel_phases(num_elements, num_samples):
    resulting_phase = np.random.rand(num_samples, num_elements)*2*np.pi
    return resulting_phase

def gains_constant_phase(channel_phases, los_phase=None, los_amp=1.):
    combined_phases = np.sum(np.exp(-1j*channel_phases), axis=-1)
    if los_phase is not None:
        combined_phases = los_amp*np.exp(1j*los_phase) + combined_phases
    return np.abs(combined_phases)**2

def rvs_ris_phases(num_elements, num_samples_slow, num_samples_fast, copula="indep"):
    if copula.startswith("comon"):
        ris_phases = np.random.rand(num_samples_fast, num_samples_slow, 1)*2*np.pi
        ris_phases = np.tile(ris_phases, (1, 1, num_elements))
    elif copula.startswith("indep"):
        ris_phases = np.random.rand(num_samples_fast, num_samples_slow, num_elements)*2*np.pi
    elif copula.startswith("counter"):
        if num_elements != 2: raise ValueError("Countermonotonic is currently only supported for 2 elements")
        ris_phases = np.random.rand(num_samples_fast, num_samples_slow, 1)*2*np.pi
        ris_phases = np.concatenate((ris_phases, 2*np.pi-ris_phases), axis=2)
    return ris_phases

def rvs_ris_phases_quant(num_elements, num_samples_slow, num_samples_fast, copula="comon", K=2):
    _choices = np.arange(K)*2*np.pi/K
    if copula.startswith("comon"):
        ris_phases = np.random.choice(_choices, (num_samples_fast, num_samples_slow, 1))
        ris_phases = np.tile(ris_phases, (1, 1, num_elements))
    elif copula.startswith("indep"):
        #ris_phases = np.random.choice(_choices, (num_samples_fast, num_samples_slow, num_elements))
        ris_phases = np.random.randint(0, K, size=(num_samples_fast, num_samples_slow, num_elements))*2*np.pi/K
    elif copula.startswith("count"):
        phase_mat = np.tile(_choices, (num_elements, 1)).T
        joint_dist = basic_rearrange(phase_mat, max)# , cost_func=_opt_func_mod2_sum)
        #print(joint_dist)#, _opt_func_mod2_sum(joint_dist, axis=1))
        #print(_opt_func_mod2_sum(joint_dist, axis=1))
        print(np.sum(joint_dist, axis=1))
        _idx = np.random.randint(0, K, (num_samples_fast, num_samples_slow))
        ris_phases = joint_dist[_idx]
    return ris_phases

def _opt_func_mod2_sum(x, *args, **kwargs):
    return np.mod(np.sum(x, *args, **kwargs), 2*np.pi)
