# Riemannian Geometry for Molecular Surface Approximation (RGMolSA)

## Introduction

Ligand-based virtual screening aims to reduce the cost and duration of drug discovery campaigns. Shape similarity can be used to screen large databases, with the goal of predicting potential new hits by comparing to molecules with known favourable properties. RGMolSA is a new alignment-free and mesh-free surface-based molecular shape descriptor derived from the mathematical theory of Riemannian geometry. The treatment of a molecule as a series of intersecting spheres allows the description of its surface geometry using the _Riemannian metric_, obtained by considering the spectrum of the Laplacian. This gives a simple vector descriptor constructed of the weighted surface area and eight non-zero eigenvalues, which capture the surface shape. The full method is described [here](https://arxiv.org/abs/2201.04230).

## In Development

RGMolSA should currently be considered a beta version under development. This initial sample runs for the supplied PDE5 inhibitor test sets (as discussed in the above paper), but is not guaranteed to work for all molecules.

## Installation

#### Dependencies
- Local Installation of [Anaconda](https://www.anaconda.com)
- [RDKit](https://www.rdkit.org/docs/Install.html)

#### Downloading the Software
Run the following in the terminal from the directory the software is to be cloned to:
```
git clone https://github.com/RPirie96/RGMolSA.git
```

Create a conda environment for the required dependencies (note this was created for MacOS but should work for other OS too)
```
conda env create -f environment.yml
```

## Running RGMolSA

The Jupyter Notebook "run_RGMolSA.ipynb" can be used to run the code for the examples provided in the paper. Note that you'll need to change the paths specified in the 1st cell to the directory the python scripts and data have been cloned to for the notebook to run.

The script "example_run.py" is a script to run the method for any dataset. It takes in 3 arguments:
- name of the file containing molecule structures
- name of the file to write the produced conformers to
- name of the csv file to write the final set of scores to

#### Data Supplied
- SVT.sdf: structure data file containing a single conformer for each of Sildenafil, Vardenafil and Tadalafil.
- X_confs_10.sdf: structure data files for Sildenafil, Vardenafil and Tadalafil, each containing 10 low energy conformers generated using RDKit. 
- X_confs_10random.sdf: structure data files for Sildenafil, Vardenafil and Tadalafil, each containing 10 random conformers generated using RDKit.
- Arginine.sdf, Diflorasone.sdf, Lymecycline.sdf and S-octylglutathione.sdf: structure data files for the 4 potential decoy molecules considered
