# Central Bank Reputation with Noise

This repository contains the numerical analysis for the paper:  


["Central Bank Reputation with Noise"](https://manuelamador.me/files/central_bank_reputation.pdf) by Manuel Amador and Christopher Phelan. 

## Requirements

The code is in Julia (version 1.12.1). Uses the following external packages: Plots and Roots.


## Instructions to run the code

The code uses multithreading, if available, for a speed up, so make sure to start julia with the ability to run multiple threads. 

Open the julia terminal in the repository folder and run the following commands to install the required packages:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

The main results are in the Jupyter notebook `main.ipynb`. This notebook generates all of the graphs and numerical results contained in the paper. The simulation runs under a minute in a standard 2025 laptop. 

The robustness checks are in the Jupyter notebook `robustness_checks.ipynb`.




