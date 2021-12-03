# Riemannian Geometry for Molecular Surface Approximation (RGMolSA)

## Introduction

## In Development

RGMolSA should currently be considered a beta version under development. This initial sample runs for the supplied PDE5 inhibitor test sets (as discussed in the above paper), but is not guaranteed to work for all molecules.

## Installation

#### Dependencies
- Local Installation of [Anaconda](https://www.anaconda.com)
- [RDKit](https://www.rdkit.org/docs/Install.html)

#### Downloading the Software

```
git clone https://github.com/RPirie96/RGMolSA.git
```

## Running RGMolSA

The Jupyter Notebook "run_RGMolSA" can be used to run the code for the examples provided in the paper. Note that you'll need to change the paths specified in the 1st cell to the directory the python scripts and data have been cloned to for the notebook to run.

#### Data Supplied
- SVT.sdf: structure data file containing a single conformer for each of Sildenafil, Vardenafil and Tadalafil.
- X_confs_10.sdf: structure data files for Sildenafil, Vardenafil and Tadalafil, each containing 10 low energy conformers generated using RDKit. 
- X_confs_10random.sdf: structure data files for Sildenafil, Vardenafil and Tadalafil, each containing 10 random conformers generated using RDKit.
