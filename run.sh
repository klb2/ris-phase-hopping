#!/usr/bin/env bash
# Run all script to generate the results presented in the paper "Artificial
# Fast Fading from Reconfigurable Surfaces Enables Ultra-Reliable
# Communications" (Eduard Jorswieck, Karl-L. Besser, Cong Sun).
#
# Copyright (C) 2021 Karl-Ludwig Besser
# License: GPLv3

echo "Running outage probability for constant phases..."
python3 constant_phases.py -N 10 50 -n 50000 --plot

echo "Running two-element example with random phases..."
python3 random_phases.py -N 2 -s 1000 -f 5000 --plot

echo "Running two-element example with quantized phases..."
python3 discrete_phases.py -N 2 -K 2 -s 1000 -f 5000 --plot
