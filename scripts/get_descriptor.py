"""
main script to return the descriptor for a single molecule

Functions:
- get_descriptor: helper function to run all modules involved in calculating RGMolSA descriptor
"""

import numpy as np
from numpy import linalg as la

from get_inputs import get_mol_info
from basesphere import get_base_sphere, get_levels, get_area, rescale_inputs, base_error
from stereo_projection import get_stereographic_projection
from b_matrix import get_b_mat
from a_matrix import get_a_mat
from utils import cut_10


def get_descriptor(mol):

    """
    Function to generate shape descriptor for a single molecule

    @param mol:
    @return: descriptor, or error type
    """

    try:
        # get centres, radii, adjacency matrix and no. atoms
        inputs = get_mol_info(mol)

        # get base sphere and re-centre on origin
        base = get_base_sphere(inputs.centres)

        # get levels within molecule
        levels = get_levels(inputs.adjacency_matrix, inputs.no_atoms, base.base_sphere)

        # get molecule area
        mol_area = get_area(
            inputs.adjacency_matrix, base.centres, inputs.no_atoms, inputs.radii
        )

        # rescale inputs so molecule has surface area equivalent to a unit sphere
        rescaled = rescale_inputs(
            mol_area.area, base.centres, inputs.radii, mol_area.lam
        )

        # error handling to account for cases where there is an atom over the north pole
        error = base_error(levels, inputs, base, rescaled)

        # perform 'piecewise stereographic projection' to move molecule into CP^n
        stereo_proj = get_stereographic_projection(
            levels, inputs, rescaled, error.centres_r
        )

        # get b matrix
        b_mat = get_b_mat()

        # get a matrix
        a_mat = get_a_mat(inputs.no_atoms, stereo_proj, error)

        # calculate final descriptor
        c_mat = np.matmul(la.inv(a_mat), b_mat)

        try:
            e_val, e_fun = la.eig(c_mat)

        except np.linalg.LinAlgError:

            if levels.no_levels <= 10:
                # remove > max level-1 spheres
                new_inputs = cut_10(inputs, error, lev_keep=levels.no_levels-1)
            else:
                # remove > level 10 spheres
                new_inputs = cut_10(inputs, error)

            # get base sphere and re-centre on origin
            base = get_base_sphere(new_inputs.centres)

            # get levels within molecule
            levels = get_levels(
                new_inputs.adjacency_matrix, new_inputs.no_atoms, base.base_sphere
            )

            # get molecule area
            mol_area = get_area(
                new_inputs.adjacency_matrix,
                base.centres,
                new_inputs.no_atoms,
                new_inputs.radii,
            )

            # rescale inputs so molecule has surface area equivalent to a unit sphere
            rescaled = rescale_inputs(
                mol_area.area, base.centres, new_inputs.radii, mol_area.lam
            )

            # error handling to account for cases where there is an atom over the north pole
            error = base_error(levels, new_inputs, base, rescaled)

            # perform piecewise stereographic projection to move molecule into CP^n
            stereo_proj = get_stereographic_projection(
                levels, new_inputs, rescaled, error.centres_r
            )

            # get b matrix
            b_mat = get_b_mat()

            # get a matrix
            a_mat = get_a_mat(new_inputs.no_atoms, stereo_proj, error)

            # calculate final descriptor
            c_mat = np.matmul(la.inv(a_mat), b_mat)
            
            e_val, e_fun = la.eig(c_mat)

        e_val = sorted(e_val)

        # change 1st eigenvalue from 0 to area to account for re-scale
        e_val[0] = 10000 / mol_area.area

        if e_val[1] < 0:  # account for final numerical instabilities
            e_val[1] = 0

        e_val = [round(num, 3) for num in e_val]

        return e_val

    except TypeError:
        return "TypeError"

    except ArithmeticError:
        return "ArithmeticError"

    except np.linalg.LinAlgError:
        return "LinAlgError"
