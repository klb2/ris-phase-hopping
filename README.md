# Artificial Fast Fading from RIS

This repository is accompanying the paper "Artificial Fast Fading from
Reconfigurable Surfaces Enables Ultra-Reliable Communications" (Eduard
Jorswieck, Karl-L. Besser, and Cong Sun. IEEE SPAWC 2021, [doi:XXX]()).

The idea is to give an interactive version of the calculations and presented
concepts to the reader. One can also change different parameters and explore
different behaviors on their own.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gl/klb2%2Fris-phase-hopping/spawc)


## File List
The following files are provided in this repository:

- [RIS Simulation.ipynb](https://mybinder.org/v2/gl/klb2%2Fris-phase-hopping/spawc?filepath=RIS%20Simulation.ipynb): Jupyter notebook that contains an interactive version of
  the simulations.
- `run.sh`: Bash script that reproduces the figures presented in the paper.
- `constant_phases.py`: Python script that runs the simulations for constant
  RIS phases.
- `random_phases.py`: Python script that runs the simulations for randomly
  varying RIS phases.
- `discrete_phases.py`: Python script that runs the simulations for randomly
  varying RIS phases with only a discrete set of possible phase values.
- `phases.py`: Python module that contains helper functions to generate the RIS
  phases according to the different scenarios.


## Usage
### Running it online
The easiest way is to use services like [Binder](https://mybinder.org/) to run
the notebook online. Simply navigate to
[https://mybinder.org/v2/gl/klb2%2Fris-phase-hopping/spawc](https://mybinder.org/v2/gl/klb2%2Fris-phase-hopping/spawc)
to run the notebooks in your browser without setting everything up locally.

### Local Installation
If you want to run it locally on your machine, Python3 and Jupyter are needed.
The present code was developed and tested with the following versions:
- Python 3.9
- Jupyter 1.0
- numpy 1.20
- scipy 1.6

Make sure you have [Python3](https://www.python.org/downloads/) installed on
your computer.
You can then install the required packages (including Jupyter) by running
```bash
pip3 install -r requirements.txt
jupyter nbextension enable --py widgetsnbextension
```
This will install all the needed packages which are listed in the requirements 
file. The second line enables the interactive controls in the Jupyter
notebooks.

Finally, you can run the Jupyter notebooks with
```bash
jupyter notebook
```

You can also recreate the figures from the paper by running
```bash
bash run.sh
```


## Acknowledgements
This research was supported in part by the Deutsche Forschungsgemeinschaft
(DFG) under grant JO 801/23-1 and by the National Natural Science Foundation of
China (NSFC) 11771056, 11871115.


## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.
