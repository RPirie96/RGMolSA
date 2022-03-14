"""
script to get the B matrix

Functions:
- sphere_integral: function to integrate polynomial over the sphere
- get_b_mat: function to get b matrix
"""

import numpy as np
from scipy.special import gamma as gam_fun


def sphere_integral(p, q, s):
    """
    function that integrates a polynomial (x^p)(y^q)(z^s) over a sphere

    @param p:
    @param q:
    @param s:
    @return: result of integration
    """

    ret = 0
    if (p % 2 - 1) * (q % 2 - 1) * (s % 2 - 1) == -1:
        beta_1 = (1 / 2) * (p + 1)
        beta_2 = (1 / 2) * (q + 1)
        beta_3 = (1 / 2) * (s + 1)
        ret = (
            2
            * gam_fun(beta_1)
            * gam_fun(beta_2)
            * gam_fun(beta_3)
            / gam_fun(beta_1 + beta_2 + beta_3)
        )

    return ret


def get_b_mat():
    """
    Code to make B matrix
    @return: b matrix (symmetric matrix arising from Rayleigh-Ritz approx. to spectrum)
    """
    grad_mat_2 = np.zeros((9, 9), dtype=float)
    eig_1 = 2
    eig_2 = 6
    grad_mat_2[0][0] = 0
    grad_mat_2[1][1] = eig_1 * sphere_integral(2, 0, 0)
    grad_mat_2[2][2] = eig_1 * sphere_integral(0, 2, 0)
    grad_mat_2[3][3] = eig_1 * sphere_integral(0, 0, 2)

    # .X^2-Y^2
    grad_mat_2[4][4] = eig_2 * (
        sphere_integral(4, 0, 0)
        - 2 * sphere_integral(2, 2, 0)
        + sphere_integral(0, 4, 0)
    )
    grad_mat_2[4][5] = eig_2 * (sphere_integral(3, 1, 0) - sphere_integral(1, 3, 0))
    grad_mat_2[4][6] = eig_2 * (sphere_integral(3, 0, 1) - sphere_integral(1, 2, 1))
    grad_mat_2[4][7] = eig_2 * (sphere_integral(2, 1, 1) - sphere_integral(0, 3, 1))
    grad_mat_2[4][8] = eig_2 * (
        3 * (sphere_integral(2, 0, 2) - sphere_integral(0, 2, 2))
        - (sphere_integral(2, 0, 0) - sphere_integral(0, 2, 0))
    )
    # .XY
    grad_mat_2[5][5] = eig_2 * (sphere_integral(2, 2, 0))
    grad_mat_2[5][6] = eig_2 * (sphere_integral(2, 1, 1))
    grad_mat_2[5][7] = eig_2 * (sphere_integral(1, 2, 1))
    grad_mat_2[5][7] = eig_2 * (3 * sphere_integral(1, 1, 2) - sphere_integral(1, 1, 0))
    # .XZ
    grad_mat_2[6][6] = eig_2 * (sphere_integral(2, 0, 2))
    grad_mat_2[6][7] = eig_2 * (sphere_integral(1, 1, 2))
    grad_mat_2[6][8] = eig_2 * (3 * sphere_integral(1, 0, 3) - sphere_integral(1, 0, 1))
    # .YZ
    grad_mat_2[7][7] = eig_2 * (sphere_integral(0, 2, 2))
    grad_mat_2[7][8] = eig_2 * (3 * sphere_integral(0, 1, 3) - sphere_integral(0, 1, 1))
    # .3Z^2-1
    grad_mat_2[8][8] = eig_2 * (
        9 * sphere_integral(0, 0, 4)
        - 6 * sphere_integral(0, 0, 2)
        + sphere_integral(0, 0, 0)
    )

    for i_spec in range(0, 9):
        for j_spec in range(0, i_spec):
            grad_mat_2[i_spec, j_spec] = grad_mat_2[j_spec, i_spec]

    return grad_mat_2
