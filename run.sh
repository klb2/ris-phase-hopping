#!/usr/bin/env bash
# Run all script to generate the results presented in the paper "Artificial
# Fast Fading from Reconfigurable Surfaces Enables Ultra-Reliable
# Communications" (Eduard Jorswieck, Karl-L. Besser, Cong Sun).
#
# Copyright (C) 2021 Karl-Ludwig Besser
# License: GPLv3

echo "Running outage probability for constant phases..."
echo "Figure 2: NLOS Scenario"
python3 constant_phases.py -a 0 -N 4 10 50 -n 1000000 --plot
echo "Figure 3: LOS Scenario with a=2"
python3 constant_phases.py -a 2 -N 4 10 50 -n 1000000 --plot

echo "Figure 4: Calculating ergodic capacities for NLOS scenario..."
python3 ergodic_capacity.py -a 0 -N {4..50} --plot

echo "Figure 5: Outage probability for NLOS sceneario..."
python3 random_phases.py -a 0 -N 4 20 50 -s 1000 -f 5000 --plot

echo "Figure 6: Outage probability log-plot..."
python3 random_phases.py -a 0 -N 4 -s 100000 -f 2000 --logplot

#echo "Running two-element example with random phases..."
#python3 random_phases.py -N 2 -s 1000 -f 5000 --plot
#
#echo "Running two-element example with quantized phases..."
#python3 discrete_phases.py -N 2 -K 2 -s 1000 -f 5000 --plot
