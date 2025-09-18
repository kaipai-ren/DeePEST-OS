# **Generic Machine Learning Potential for Accelerating Transition State Search in Organic Synthesis**

[![License](https://img.shields.io/github/license/kaipai-ren/DeePEST-OS)](https://img.shields.io/github/license/kaipai-ren/DeePEST-OS) [![ChemRxiv](https://img.shields.io/badge/ChemRxiv-Preprint-orange)](https://chemrxiv.org/engage/chemrxiv/article-details/684161351a8f9bdab5d606ae)

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Instructions for use](#instructions-for-use)
- [Citation](#citation)
- [Contact](#contact)

## Overview

This repository includes the structures of organic reaction systems discussed in the article [**Generic Machine Learning Potential for Accelerating Transition State Search in Organic Synthesis**](https://chemrxiv.org/engage/chemrxiv/article-details/684161351a8f9bdab5d606ae) as well as the corresponding demo code for transition state structure optimizations and IRC calculations using the **D**eep learning-based molecular **P**otential **E**nergy **S**urface prediction **T**ool for **O**rganic **S**ynthesis (DeePEST-OS).

## Repo Contents

**[DeePEST-OS](https://github.com/kaipai-ren/DeePEST-OS)**  
│─ [dataset](./dataset): files in XYZ format of transition state initial guesses for reactions, produced by the `GENiniTS-RS` software.  
│  ├─ [1000_external_test_set](./dataset/1000_external_test_set): 1k initial structures of external test reactions in this work.  
│  └─ [conformational_isomer](./dataset/conformational_isomer): conformation isomers in the transition state conformational isomer screening case.  
│  └─ [cross-dataset_validation_of_DeePEST-OS-T1x](./dataset/cross-dataset_validation_of_DeePEST-OS-T1x):  cross validation dataset of MACE_delta model trained on Transition1x database.  
│  └─ [multi-step_organic_reactions](./dataset/multi-step_organic_reactions): intermediate and transition state initial structures in multi-step organic reaction retrosynthesis case.  
│─ [examples](./examples): sample scripts for model training and inference are here.  
│  ├─ [ts_and_irc](./examples/ts_and_irc): demo for transition state optimization and IRC calculation.  
│  └─ [model_training](./examples/model_training): demo for training machine learning potential model in this work.  
│─ [models](./models): all the relevant model files in this work.  
│  ├─ [MACE](./models/MACE): MACE model trained without delta learning strategy.  
│  └─ [MACE_deltaL](./models/MACE_deltaL): MACE model trained with delta learning strategy.  
│  └─ [PaiNN](./models/PaiNN): PaiNN model trained without delta learning strategy.  
│  └─ [DeePEST-OS-T1x](./models/DeePEST-OS-T1x): MACE model trained one Transition1x dataset..  
│─ [requirements](./requirements): python packages and their versions in the virtual environment required to run the RMLP models.

## System Requirements

### Hardware Requirements

The working examples in this repository require a standard computer with CPU, NVIDIA GPU and enough RAM to support the operations defined by a user. When the computer doesn't have an NVIDIA GPU, the machine learning potential model can only be inferred on the CPU, which makes the model less efficient. We recommend a computer with the following specs:

```
CPU: 4+ cores, 3.3+ GHz/core
RAM: 16+ GB
GPU: NVIDIA GPU with 4+ GB memory
```

### Software Requirements

This code can be run on **Linux** system using a **Conda** environment.


## Installation Guide

Please install the conda environment according to one of the following methods.

- **Installation option 1**

   Rebuilding the conda environment using dependency files ([deepest_os.txt](./requirements/deepest_os.txt)).

  After downloading this repository, navigate to the [requirements folder](./requirements) in a terminal (execution on a Linux system is recommended) and run the following command to install the virtual environment:

  ```
  conda create --name deepest_os --file deepest_os.txt
  conda activate deepest_os
  ```

  The virtual environment installation should be completed within tens of minutes.

- **Installation option 2**

  Rebuild the conda environment by downloading the [virtual environment compressed package](https://zenodo.org/records/17141212).

  Please follow the commands below to install the conda environment:
  
  ```
  cd /path/to/your/conda/environment
  mkdir deepest_os 
  cd deepest_os 
  wget -O deepest_os.tar.gz "https://zenodo.org/records/17141212/files/deepest_os.tar.gz?download=1"
  tar -zxvf deepest_os.tar.gz
  conda activate deepest_os
  ```

## Demo

The Demo folder contains example scripts for machine learning potential model training and inference (transition state optimizations).

- [ts_and_irc](./examples/ts_and_irc): example for transition state optimizations and IRC calculations.

  > On a laptop with an AMD Ryzen 9 7945HX processor, NVIDIA GeForce RTX 4060, and 48 GB of RAM, running `ts_opt.ipynb` allows transition state optimization of each XYZ structure to finish within ten seconds. For instructions on how to run the .ipynb tutorial script, see the [**Instructions for use **](#instructions-for-use) section of this README.
  >
  > The input files for transition state optimization and IRC calculations are located in the `inputs` folder within the same directory as the `ts_opt.ipynb` file. The xyz file containing the initial guess for the transition state structures are provided. The output files from the script are located in the `outputs` folder and include the complete transition state structure obtained through the machine learning potential-driven search, the IRC path, the optimized reactant and product structures, the optimized trajectory file, an image of the IRC path, and a GIF of the reaction process.

- [model_training](./examples/model_training): example for training a MACE model.

  > Train the model by running `run_train.py`. On a laptop with an AMD Ryzen 9 7945HX processor, NVIDIA GeForce RTX 4060, and 48 GB of RAM, training the model on the example dataset (`demo.xyz`) takes roughly 40 seconds per epoch (batch_size=10).
  >
  > The input file for model training is `demo_train.xyz`, and the test set of the model is `demo_test.xyz`. During the training process, a `log` folder containing the training process log file and the final model file and a `ckp` folder containing the checkpoint files will be generated.

## Instructions for use

After configuring the necessary virtual environment, run the demo scripts as follows:

- For transition state optimizations and IRC calculations:

  1. `conda activate deepest_os`
  2. `cd /path/to/the/example/jupyter/notebook`
  3. `jupyter lab` (pip install jupyterlab if not installed)
  4. run the cells in **ts_opt.ipynb** as instructed.

- For model training:

  1. `conda activate deepest_os`
  2. `cd /path/to/the/example/python/script`
  3. split the full demo.xyz dataset by `python split_dataset.py`
  4. `python run_train --config=config.yaml`

## Citation

```
@article{ren2025deepest,
  title={DeePEST-OS: A Generic Machine Learning Potential for Accelerating Transition State Search in Organic Synthesis},
  author={Ren, Kaipai and Tang, Kun and Zhao, Yujing and Zhang, Lei and Du, Jian and Meng, Qingwei and Liu, Qilei},
  year={2025},
  journal = {ChemRxiv}
}
```

## Contact

Please contact us ([liuqilei@dlut.edu.cn](mailto:liuqilei@dlut.edu.cn)) if you have any question about our implementation.


