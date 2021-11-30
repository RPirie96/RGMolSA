# Riemannian Geometry for Molecular Surface Approximation (RGMolSA)

## Introduction

1. Get Inputs
2. find base sphere and number of levels
3. calc. area and rescale
4. error handling
5. stereographic projection
6. get b matrix
7. get a matrix
8. combine the A and B matrices, extract eigenvalues and set 1st value to surface area
9. compute inverse Bray-Curtis distance

## In Development

RGMolSA should currently be considered a beta version under development. This initial sample runs for the supplied PDE5 inhibitor test sets (as discussed in the above paper), but is not guaranteed to work for all molecules.

## Installation

#### Dependencies

conda install -c conda-forge chembl_structure_pipeline

#### Downloading the Software

## Running RGMolSA

#### Getting Started

#### Script Directory
- **data\_filters.py**: run this script before computing descriptors to prepare the dataset. Each function takes a list of molecules and a list of their IDs (optional) and returns the updated contents as a named tuple. 
  - _macrocycle\_filter_: remove macrocycles (limitation of method is that these cannot be described).
  - _small\_filter_: remove molecules with  <6 heavy atoms (molecules smaller than benzene unlikely to be of interest in small molecule drug discovery)
  - _salt\_filter_: remove salts to return the parent molecule (using the [ChEMBL Structure Pipeline](https://github.com/chembl/ChEMBL_Structure_Pipeline))
  - _druglike\_filter_: remove non-druglike molecules (defined as molecules containing elements other than H, C, N, O, F, P, S, Cl, Br, I or more than 7 B). 
  - _filter\_dataset_: helper function to run all of the above filters. 

- **get\_descriptor.py**: main script to run the software, computes the descriptor associated with the molecular surface. 
- **plot\_mol.py**: 
  - _spheres\_plot_: plot the molecule as a series of intersecting spheres post-ring replacement. 
  - _cp\_plot_:plot the molecule after mapping into $\mathbb{CP}^1$. 
The remaining scripts should not need accessed to run the software (other than to use the plotting module). Details of what each contains are included below to map to the stages in descriptor calculation:
- **get\_inputs.py**: takes the molecule in SDF format and retrieves the starting representation (no. atoms, radii, 3D centres and adjacency matrix). Carries out replacement of the rings with a single sphere. Returns a named tuple containing the no. atoms, radii, 3D centres and adjacency matrix. Use the outputs of this to plot the 3D structure of the molecule. 
- **basesphere.py**: 
  -  _get\_base\_sphere_: find the centroid of the molecule to use as the starting or "base" sphere. 
  -  _get\_levels_: produce the map from the base sphere to step through the rest of the molecule one step or level out from the base sphere at a time. 
  -  _get\_area_: calculate the surface area of the molecule, taken as the area of each sphere - the bits where two spheres intersect. 
  -  _rescale\_inputs_: rescale the properties associated with the surface such that it has surface area equal to that of a unit sphere.
  -  _base_error_: handles cases where there is an atom over the "north pole" of the base sphere which creates issues with the stereographic projection to $\mathbb{CP}^1$.
  
- **stereo\_projection.py**: completes the piecewise stereographic projection from real space to $\mathbb{CP}^1$.
- **b\_matrix.py**:computes the B matrix associated with the surface.
- **a\_matrix.py**: computes the A matrix associated with the surface
- **utils.py**:contains helper functions for the stereographic projection, the integration functions to compute the A matrix and the function to compute the score between two molecules.

#### Data Supplied
- SVT.sdf: structure data file containing a single conformer for each of Sildenafil, Vardenafil and Tadalafil.
- X_confs_10.sdf: structure data files for Sildenafil, Vardenafil and Tadalafil, each containing 10 low energy conformers generated using RDKit. 
- X_confs_10random.sdf: structure data files for Sildenafil, Vardenafil and Tadalafil, each containing 10 random conformers generated using RDKit.
