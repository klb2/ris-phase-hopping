"""Simulation functions for phases

This module contains functions to generate the RIS phases according to the
different scenarios.


Copyright (C) 2021 Karl-Ludwig Besser

This program is used in the article:
Eduard Jorswieck, Karl-Ludwig Besser, and Cong Sun, "Artificial Fast Fading
from Reconfigurable Surfaces Enables Ultra-Reliable Communications", IEEE
International Workshop on Signal Processing Advances in Wireless Communications
(SPAWC), 2021.

License:
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.
See the GNU General Public License for more details.

Author: Karl-Ludwig Besser, Technische Universität Braunschweig
"""
__author__ = "Karl-Ludwig Besser"
__copyright__ = "Copyright (C) 2021 Karl-Ludwig Besser"
__credits__ = ["Karl-Ludwig Besser", "Eduard A. Jorswieck"]
__license__ = "GPLv3"
__version__ = "1.0"

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def rvs_channel_phases(num_elements, num_samples):
    resulting_phase = np.random.rand(num_samples, num_elements)*2*np.pi
    return resulting_phase

def gains_constant_phase(channel_phases):
    combined_gains = np.abs(np.sum(np.exp(-1j*channel_phases), axis=-1))**2
    return combined_gains

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
        ris_phases = np.random.choice(_choices, (num_samples_fast, num_samples_slow, num_elements))
    return ris_phases
