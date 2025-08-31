# DeePEST-OS: A Generic Machine Learning Potential for Accelerating Transition State Search in Organic Synthesis

## Introduction

This repository includes the structures of organic reaction systems discussed in the article "**DeePEST-OS: A Generic Machine Learning Potential for Accelerating Transition State Search in Organic Synthesis**,"(https://chemrxiv.org/engage/chemrxiv/article-details/684161351a8f9bdab5d606ae) as well as the corresponding code for transition state structure optimization and energy barrier prediction using the Deep learning-based molecular Potential Energy Surface prediction Tool for Organic Synthesis(DeePEST-OS).

The contents of each folder are as follows:

1. **dataset**: 1000 initial guess geometries of transition states for external validation.
2. **example**: Workfolder of transition state optimization and energy barrier predictio by DeePEST-OS model.
3. **models**: The DeePEST-OS model in the article.
4. **ts_opt.py**: Script for transition state optimization and energy barrier prediction driven by DeePEST-OS model.
5. **environment.yml and packages.txt**: The required environment information for running DeePEST-OS model.
   
## Installation environment
After downloading the DeePEST-OS.zip file and extracting it, navigate to the folder in a terminal (execution on a Linux system is recommended) and run the following command to install the environment:

```
conda create --name deepest_os
conda env update --name deepest_os --file environment.yml
conda install --name deepest_os --file packages.txt
```


## Usage tutorial

After downloading the repository using git clone or similar commands, move to the generated directory and run the following:

```
python ts_opt.py
```

This command will use the DeePEST-OS model to optimize the initial guess of transition state structures and predict energy barrier in `./example/input` folder , and output to `./example/output`.

Users can modify the input file path and output file path in ts_opt.py.
