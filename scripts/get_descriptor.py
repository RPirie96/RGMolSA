# main script to return the descriptor for a single molecule

import numpy as np
from numpy import linalg as la

from get_inputs import get_mol_info
from basesphere import get_base_sphere, get_levels, get_area, rescale_inputs, base_error
from stereo_projection import get_stereographic_projection
from b_matrix import get_b_mat
from a_matrix import get_a_mat

def get_descriptor(mol):
    """ Function to generate shape descriptor for a single molecule"""

    # get centres, radii, adjacency matrix and no. atoms
    inputs = get_mol_info(mol)

    # get base sphere and re-centre on origin
    base = get_base_sphere(inputs.centres)

    # get levels within molecule
    levels = get_levels(inputs.adjacency_matrix, inputs.no_atoms, base.base_sphere)

    # get molecule area
    mol_area = get_area(inputs.adjacency_matrix, base.centres, inputs.no_atoms, inputs.radii)

    # rescale inputs so molecule has surface area equivalent to a unit sphere
    rescaled = rescale_inputs(mol_area.area, base.centres, inputs.radii, mol_area.lam)

    # error handling to account for cases where there is an atom over the north pole
    error = base_error(levels, inputs, base, rescaled)

    # perform 'piecewise stereographic projection' to move molecule into CP^n
    stereo_proj = get_stereographic_projection(levels, inputs, rescaled, error.centres_r)

    # get b matrix
    b_mat = get_b_mat()

    # get a matrix
    a_mat = get_a_mat(inputs.no_atoms, stereo_proj, error)

    # calculate final descriptor
    c_mat = np.matmul(la.inv(a_mat), b_mat)
    e_val, w = la.eig(c_mat)
    e_val = sorted(e_val)

    e_val[0] = 10000 / mol_area.area  # change 1st eigenvalue from 0 to area to account for re-scale

    e_val = [round(num, 3) for num in e_val]

    return e_val
