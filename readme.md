# Central Bank Reputation with Noise

This repository contains the numerical analysis for the paper:  


["Central Bank Reputation with Noise"](https://manuelamador.me/files/central_bank_reputation.pdf) by Manuel Amador and Christopher Phelan. 

## Requirements

The code is in Julia (version 1.12.1). Uses the following external packages: Plots and Roots.


## Instructions to run the code

The code uses multithreading, if available, for a speed up, so make sure to start julia with the ability to run multiple threads. To let Julia decide the number of threads to use, start with the option `-t auto`:

```bash
julia -t auto
```

Open the julia terminal in the repository folder and run the following commands to install the required packages:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

To run the simulation, execute the script `main.jl`:
```julia
include("main.jl")
```

The script runs in under a minute in a standard laptop. The script will generate all of the graphs contained in the paper and will save the series in the `results` folder.


