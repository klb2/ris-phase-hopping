#!/usr/bin/env bash
# Run all script to generate the results presented in the paper "Reconfigurable
# Intelligent Surface Phase Hopping for Ultra-Reliable Communications" (Karl-L.
# Besser, Eduard Jorswieck).
#
# Copyright (C) 2021 Karl-Ludwig Besser
# License: GPLv3

echo "Figure 2: Outage probability constant phases NLOS Scenario"
python3 constant_phases.py -a 0 -N 4 10 50 -n 1000000 --plot

echo "Figure 3: Outage probability constant phases LOS Scenario with a=2"
python3 constant_phases.py -a 2 -N 4 10 50 -n 1000000 --plot

echo "Figure 4: Calculating ergodic capacities for NLOS scenario..."
python3 ergodic_capacity.py -a 0 -N {4..50} --plot

echo "Figure 5: Outage probability for NLOS sceneario..."
#python3 random_phases.py -a 0 -N 4 20 50 -s 1000 -f 5000 --plot
python3 random_phases.py -a 0 -N 20 -p .1 .5 .9 -s 2000 -f 10000 -b 1000 --plot

echo "Figure 6: Outage probability log-plot..."
python3 random_phases.py -a 0 -N 4 -s 100000 -f 2000 --logplot

echo "Figure 7: Calculating ergodic capacities for LOS scenario..."
python3 ergodic_capacity.py -a 0 2 4 -N {4..30} --plot

echo "Figure 8: Outage probability for LOS sceneario..."
python3 random_phases.py -a 2 -N 4 10 50 -s 1000 -f 5000 --plot
