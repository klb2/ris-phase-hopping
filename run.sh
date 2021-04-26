#!/usr/bin/env bash
# Run all script to generate the results presented in the paper "Artificial
# Fast Fading from Reconfigurable Surfaces Enables Ultra-Reliable
# Communications" (Eduard Jorswieck, Karl-L. Besser, Cong Sun).
#
# Copyright (C) 2021 Karl-Ludwig Besser
# License: GPLv3

echo "Running outage probability for constant phases..."
python3 constant_phases.py -N 10 50 --plot

echo "Running two-element example with random phases..."
python3 

echo "Running two-element example with quantized phases..."
python3 
