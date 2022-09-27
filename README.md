# FootSim
Simulating peripheral tactile responses from the sole of the foot.

## Installation
The footsim package requires Python 3.6 or higher to run. It also requires *numpy*, *scipy*, *skikit-image*, *numba*, and *matplotlib*. Additionally, *holoviews* is required to use the simulation's inbuilt plotting functions.

If using conda for package management, the following command creates a new environment *fs* with all dependencies installed:
```conda env create -f environment.yml```

To install the package via a symlink to the git directory, use ```python setup.py develop```.

## Using the package
Examples of how to use the model and its plotting functions are contained in an iPython notebook, see *Examples.ipynb*.
