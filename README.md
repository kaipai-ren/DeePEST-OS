# DeePEST-OS: A Generic Machine Learning Potential for Accelerating Transition State Search in Organic Synthesis

## Introduction

This repository includes the structures of organic reaction systems discussed in the article "**DeePEST-OS: A Generic Machine Learning Potential for Accelerating Transition State Search in Organic Synthesis**,"(https://chemrxiv.org/engage/chemrxiv/article-details/684161351a8f9bdab5d606ae) as well as the corresponding code for transition state structure optimization and energy barrier prediction using the Deep learning-based molecular Potential Energy Surface prediction Tool for Organic Synthesis(DeePEST-OS).

The contents of each folder are as follows:

1. **dataset**: 1000 initial guess geometries of transition states for external validation.
2. **example**: Workfolder of transition state optimization and energy barrier predictio by DeePEST-OS model.
3. **models**: The DeePEST-OS models in the article, including MACE_deltaL, MACE, PaiNN.
4. **neuralneb**: Dependency modules of PaiNN model.
5. **ts_opt.py**: Script for transition state optimization and energy barrier prediction driven by DeePEST-OS models.

## Required modules

- torch
- numpy
- xtb
- mace-torch
- ase
- sella
- matplotlib
- natsort
- configparser
- os
- time
- traceback
- argparse
- x3dase

## Usage tutorial

After downloading the repository using git clone or similar commands, move to the generated directory and run the following:

```
python ts_opt.py --model_name='MACE_deltaL'
```

This command will use the MACE_deltaL model to optimize the initial guess of transition state structures and predict energy barrier in `./example/input` folder , and output to `./example/output`.

Other arguments:

```
--input_path
```

Type: str. Specifies the XYZ format geometry path for the input.

```
--output_path
```

Type: str. Specifies the output file path.

```
--model_path
```

Type: str. Specifies the DeePEST-OS model file path.

