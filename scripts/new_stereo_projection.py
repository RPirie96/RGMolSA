"""
script to compute the stereographic projection of the molecule

Functions:
- get_stereographic_projection: piecewise stereographic projection of the molecule into CP^n
"""

from collections import namedtuple
import math
import numpy as np
from numpy import linalg as la

from utils import get_chain, get_m_rot, alpha_coefficient, beta_coefficient, t_circle


def get_stereographic_projection(inputs):
    """
    Function to return the piecewise stereographic projection of the molecule into CP^n
    @param inputs:
    @return: named tuple with stereographic projection constructs
    """

    # unpack tuple
    no_levels = inputs.no_levels
    level_mat = inputs.level_mat
    no_atoms = inputs.no_atoms
    adjacency_matrix = inputs.adjacency_matrix
    radii_r = inputs.radii_r
    lam_r = inputs.lam_r
    centres_r = inputs.centres_r

    # set up empty matrices
    alpha_mat = np.zeros(
        (no_levels + 1, no_atoms), dtype=float
    )  # matrix of alpha coefficients
    beta_mat = np.zeros(
        (no_levels + 1, no_atoms), dtype=complex
    )  # matrix of beta coefficients

    disc_radii = np.zeros(
        (no_levels + 1, no_atoms), dtype=float
    )  # matrix of disc radii
    rel_radii = np.zeros(
        (no_levels + 1, no_atoms), dtype=float
    )  # matrix of relative radii

    com_plan_cent_rel = np.zeros(
        (no_levels + 1, no_atoms), dtype=complex
    )  # the relative centres
    com_plan_rad_rel = np.zeros(
        (no_levels + 1, no_atoms), dtype=float
    )  # the relative radii

    for i in range(1, no_levels + 1):
        level = i
        for k in range(0, no_atoms):
            if level_mat[i][k] > 0:
                sphere = k
                chain_s = get_chain(
                    no_atoms, level_mat, adjacency_matrix, sphere, level
                )

                # Set up vectors of relative centres
                rel_cent = np.zeros((level + 1, 3), float)
                for q in range(1, level + 1):
                    s_p_rel = int(chain_s[q])
                    s_p_rel_mo = int(chain_s[q - 1])
                    rel_cent[q] = centres_r[s_p_rel] - centres_r[s_p_rel_mo]
                    norm = math.sqrt(
                        (rel_cent[q][0] ** 2)
                        + (rel_cent[q][1] ** 2)
                        + (rel_cent[q][2] ** 2)
                    )
                    rel_cent[q] = rel_cent[q] / norm

                # Set up transformed vectors
                for q in range(2, level + 1):
                    rot_mat = get_m_rot(
                        [rel_cent[q - 1][0], rel_cent[q - 1][1], rel_cent[q - 1][2]]
                    )
                    inv_mat = la.inv(rot_mat)
                    for l in range(q, level + 1):
                        rel_vec = (
                            np.dot(
                                inv_mat,
                                np.array(
                                    [
                                        [rel_cent[l][0]],
                                        [rel_cent[l][1]],
                                        [rel_cent[l][2]],
                                    ]
                                ),
                            )
                        ).reshape(1, 3)
                        rel_cent[l] = rel_vec

                h_ght = 2 * radii_r[s_p_rel_mo] - abs(
                    radii_r[s_p_rel_mo] - lam_r[s_p_rel_mo][s_p_rel]
                )
                r_l = math.sqrt(h_ght / ((2 * radii_r[s_p_rel_mo]) - h_ght))

                h_ght_rel = 2 * radii_r[s_p_rel] - abs(
                    radii_r[s_p_rel] - lam_r[s_p_rel][s_p_rel_mo]
                )
                r_rel = math.sqrt(
                    1 / (h_ght_rel / ((2 * radii_r[s_p_rel]) - h_ght_rel))
                )

                alpha_mat[i][k] = alpha_coefficient(
                    [rel_cent[level][0], rel_cent[level][1], rel_cent[level][2]]
                )
                beta_mat[i][k] = beta_coefficient(
                    [rel_cent[level][0], rel_cent[level][1], rel_cent[level][2]]
                )
                disc_radii[i][k] = r_l
                rel_radii[i][k] = r_rel

                alpha = alpha_mat[i][k]
                beta = beta_mat[i][k]
                gamma = -np.conj(beta)
                delta = alpha

                com_plan_cent_rel[i][k] = t_circle(alpha, beta, gamma, delta, 0, r_l)[0]
                com_plan_rad_rel[i][k] = t_circle(alpha, beta, gamma, delta, 0, r_l)[1]

    # Code to find centres of spheres in complex plane
    com_plan_cent = np.zeros((no_levels + 1, no_atoms), dtype=complex)
    com_plan_rad = np.zeros((no_levels + 1, no_atoms), dtype=float)

    for i in range(1, no_levels + 1):
        level = i
        for k in range(0, no_atoms):
            if level_mat[i][k] > 0:
                sphere = k
                chain_s = get_chain(
                    no_atoms, level_mat, adjacency_matrix, sphere, level
                )
                mobius = np.array([[1, 0], [0, 1]])
                q = level

                while q > 1:
                    sp_mo = int(chain_s[q - 1])

                    alpha = alpha_mat[q - 1][sp_mo]
                    beta = beta_mat[q - 1][sp_mo]

                    mat = np.array([[alpha, beta], [-np.conj(beta), alpha]])
                    s = disc_radii[q - 1][sp_mo] / rel_radii[q - 1][sp_mo]
                    s_mat = np.array([[math.sqrt(s), 0], [0, math.sqrt(1 / s)]])
                    mobius = mat.dot(s_mat.dot(mobius))

                    q = q - 1

                alpha = mobius[0][0]
                beta = mobius[0][1]
                gamma = mobius[1][0]
                delta = mobius[1][1]

                com_plan_cent[i][k] = t_circle(
                    alpha,
                    beta,
                    gamma,
                    delta,
                    com_plan_cent_rel[i][k],
                    com_plan_rad_rel[i][k],
                )[0]
                com_plan_rad[i][k] = t_circle(
                    alpha,
                    beta,
                    gamma,
                    delta,
                    com_plan_cent_rel[i][k],
                    com_plan_rad_rel[i][k],
                )[1]

    d_0_mat_t = np.zeros((no_levels + 1, no_atoms), dtype=float)
    d_1_mat_t = np.zeros((no_levels + 1, no_atoms), dtype=complex)
    d_2_mat_t = np.zeros((no_levels + 1, no_atoms), dtype=float)

    for i in range(0, no_levels + 1):
        level = i
        for k in range(0, no_atoms):
            if level_mat[i][k] > 0:
                sphere = k
                chain_s = get_chain(
                    no_atoms, level_mat, adjacency_matrix, sphere, level
                )
                mobius = np.array([[1, 0], [0, 1]])
                q = level

                while q > 0:
                    sp = int(chain_s[q])

                    alpha = alpha_mat[q][sp]
                    beta = beta_mat[q][sp]

                    mat = np.array([[alpha, beta], [-np.conj(beta), alpha]])
                    s = disc_radii[q][sp] / rel_radii[q][sp]
                    s_mat = np.array([[math.sqrt(s), 0], [0, math.sqrt(1 / s)]])
                    mobius = mat.dot(s_mat.dot(mobius))

                    q = q - 1

                inv_mobius = la.inv(mobius)
                alpha = inv_mobius[0][0]
                beta = inv_mobius[0][1]
                gamma = inv_mobius[1][0]
                delta = inv_mobius[1][1]

                d_0_mat_t[i][k] = (
                    2 * (radii_r[k] ** 2) / ((abs(alpha) ** 2 + abs(gamma) ** 2) ** 2)
                )
                d_1_mat_t[i][k] = -(
                    (np.conj(alpha) * beta) + (np.conj(gamma) * delta)
                ) / (abs(alpha) ** 2 + abs(gamma) ** 2)
                d_2_mat_t[i][k] = 1 / ((abs(alpha) ** 2 + abs(gamma) ** 2) ** 2)

    sgp = namedtuple(
        "sgp", ["com_plan_cent", "com_plan_rad", "d_0_mat_t", "d_1_mat_t", "d_2_mat_t"]
    )

    return sgp(
        com_plan_cent=com_plan_cent,
        com_plan_rad=com_plan_rad,
        d_0_mat_t=d_0_mat_t,
        d_1_mat_t=d_1_mat_t,
        d_2_mat_t=d_2_mat_t,
    )