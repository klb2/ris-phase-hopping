#!/usr/bin/env bash
# Run all script to generate the results presented in the paper "Reconfigurable
# Intelligent Surface Phase Hopping for Ultra-Reliable Communications" (Karl-L.
# Besser, Eduard Jorswieck).
#
# Copyright (C) 2021 Karl-Ludwig Besser
# License: GPLv3

echo "Figure 3: Calculating ergodic capacities for NLOS scenario..."
python3 ergodic_capacity.py -a 0 -N {4..50} --plot

echo "Figure 4: Outage probability for NLOS sceneario..."
python3 random_phases.py -a 0 -N 20 -p .1 .5 .9 -s 2000 -f 10000 -b 1000 --plot

echo "Figure 5: Epsilon-outage capacity..."
python3 eps_capacity.py -a 0 -N 20 50 -p .1 .5 .9 --plot

echo "Figure 6: Calculating ergodic capacities for LOS scenario..."
python3 ergodic_capacity.py -a 0 2 4 -N {4..30} --plot

echo "Figure 7: Outage probability for LOS sceneario..."
python3 random_phases.py -a 3 -N 20 -p .1 .5 .9 -s 2000 -f 100000 -b 100 --plot --parallel

echo "Figure 8 and 9: Outage probability for N=20 with quantized phases..."
python3 discrete_phases.py -N 20 -K 2 3 10 -p .1 .5 .9 -f 100000 -s 2000 -b 20 --plot --export --parallel

echo "Figure 10: Static RIS Phases"
python3 constant_phases.py -N 20 -a 0. -p .1 .5 .9 -s 1000000 --plot --export

echo "Figure 11: Comparison of Phase Hopping, Static, and Perfect Phase Adjustments"
python3 comparison.py -N 20 -p .5 --plot --export
#python3 comparison.py -N 20 -p .1 .5 .9 --plot --export
