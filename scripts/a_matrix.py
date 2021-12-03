# script to get the A Matrix
import numpy as np
from numpy import linalg as la
import utils


def get_a_mat(no_atoms, sgp, error):
    """
    Code to make A matrix
    """
    com_plan_cent = sgp.com_plan_cent
    com_plan_rad = sgp.com_plan_rad
    d_0_mat_t = sgp.d_0_mat_t
    d_1_mat_t = sgp.d_1_mat_t
    d_2_mat_t = sgp.d_2_mat_t
    sphere_levels_vec = error.sphere_levels_vec
    next_level = error.next_level

    # Computing the spherical transformations
    transformations = []
    transformed_radii = []
    for sphere in range(0, no_atoms):

        level = sphere_levels_vec[sphere]
        if level == 0:
            radius = 10
            d_inv = ([[1, 0], [0, 1]])
            com_plan_rad[0][sphere] = 10
        else:
            d = utils.transform_v2(com_plan_cent[level][sphere], com_plan_rad[level][sphere])  # maps circle to origin
            radius = utils.t_circle(d[0][0], d[0][1], d[1][0], d[1][1], com_plan_cent[level][sphere], com_plan_rad[level][sphere])[1]
            d_inv = la.inv(d)

        transformations.append(d_inv)
        transformed_radii.append(radius)

    a_matrix = np.zeros((9, 9), dtype=complex)

    for sphere in range(0, no_atoms):

        level = sphere_levels_vec[sphere]

        radius = transformed_radii[sphere]
        d_inv = transformations[sphere]

        a = d_1_mat_t[level][sphere]
        b = d_2_mat_t[level][sphere]
        c = d_0_mat_t[level][sphere]

        alpha = d_inv[0][0]
        beta = d_inv[0][1]

        k_2 = abs(alpha + (a * np.conj(beta))) ** 2 + (b * abs(beta) ** 2)
        k_1 = ((alpha + a * np.conj(beta)) * np.conj(beta - a * np.conj(alpha))) - b * alpha * np.conj(beta)
        k_0 = abs(beta - a * np.conj(alpha)) ** 2 + b * abs(alpha) ** 2

        k_1 = k_1 / k_2
        k_0 = k_0 / k_2
        c = c / (k_2 ** 2)

        a = -np.conj(k_1)
        b = k_0 - abs(a) ** 2

        induced_1 = np.array([
            [(alpha ** 2 - np.conj(beta) ** 2).real, (alpha ** 2 + np.conj(beta) ** 2).imag,
             -(2 * alpha * np.conj(beta)).real],
            [-(alpha ** 2 - np.conj(beta) ** 2).imag, (alpha ** 2 + np.conj(beta) ** 2).real,
             (2 * alpha * np.conj(beta)).imag],
            [alpha * beta + np.conj(alpha * beta), 2 * (alpha * beta).imag, abs(alpha) ** 2 - abs(beta) ** 2]
        ])

        b_r = beta.real
        b_i = beta.imag

        induced_2 = np.array([
            [alpha ** 4 + b_i ** 4 + b_r ** 4 - 6 * b_i ** 2 * b_r ** 2, -2 * b_i * b_r * (b_i ** 2 - b_r ** 2),
             -alpha * b_r * (alpha ** 2 + 3 * b_i ** 2 - b_r ** 2), alpha * b_i * (alpha ** 2 - b_i ** 2 + 3 * b_r ** 2),
             6 * alpha ** 2 * (b_r ** 2 - b_i ** 2)],
            [-8 * b_i * b_r * (b_i ** 2 - b_r ** 2), alpha ** 4 - b_i ** 4 - b_r ** 4 + 6 * b_i ** 2 * b_r ** 2,
             -2 * alpha * b_i * (alpha ** 2 + b_i ** 2 - 3 * b_r ** 2),
             -2 * alpha * b_r * (alpha ** 2 - 3 * b_i ** 2 + b_r ** 2), 24 * alpha ** 2 * b_i * b_r],
            [4 * alpha * b_r * (alpha ** 2 + 3 * b_i ** 2 - b_r ** 2),
             2 * alpha * b_i * (alpha ** 2 - 3 * b_r ** 2 + b_i ** 2),
             alpha ** 4 - 6 * alpha ** 2 * b_r ** 2 - b_i ** 4 + b_r ** 4,
             2 * b_i * b_r * (b_i ** 2 + b_r ** 2 - 3 * alpha ** 2), 12 * alpha * b_r * (b_i ** 2 + b_r ** 2 - alpha ** 2)],
            [-4 * alpha * b_i * (alpha ** 2 - b_i ** 2 + 3 * b_r ** 2),
             2 * alpha * b_r * (alpha ** 2 + b_r ** 2 - 3 * b_i ** 2),
             2 * b_i * b_r * (b_i ** 2 + b_r ** 2 - 3 * alpha ** 2),
             alpha ** 4 + b_i ** 4 - b_r ** 4 - 6 * alpha ** 2 * b_i ** 2,
             12 * alpha * b_i * (b_i ** 2 + b_r ** 2 - alpha ** 2)],
            [-2 * alpha ** 2 * (b_i ** 2 - b_r ** 2), 2 * alpha ** 2 * b_i * b_r,
             -alpha * b_r * (b_i ** 2 + b_r ** 2 - alpha ** 2), -alpha * b_i * (b_i ** 2 + b_r ** 2 - alpha ** 2),
             6 * b_i ** 4 + 12 * b_i ** 2 * b_r ** 2 + 6 * b_r ** 4 - 6 * (b_i ** 2 + b_r ** 2) + 1]
        ])

        # degree 0
        v_val = utils.vol_integral(a, b, c, radius)

        # degree 1

        x_val = utils.x_integral(a, b, c, radius)
        y_val = -utils.x_integral(1j * a, b, c, radius)
        z_val = utils.z_integral(a, b, c, radius)

        # degree 2

        xx_val = utils.xx_integral(a, b, c, radius)
        yy_val = utils.xx_integral(1j * a, b, c, radius)
        zz_val = v_val - xx_val - yy_val
        xy_val = utils.xy_integral(a, b, c, radius)
        xz_val = utils.xz_integral(a, b, c, radius)
        yz_val = -utils.xz_integral(1j * a, b, c, radius)

        # degree 3

        xxx_val = utils.xxx_integral(a, b, c, radius)
        xxy_val = utils.xxy_integral(a, b, c, radius)
        xxz_val = utils.xxz_integral(a, b, c, radius)
        xyy_val = utils.xxy_integral(1j * a, b, c, radius)
        xyz_val = utils.xyz_integral(a, b, c, radius)
        yyy_val = -utils.xxx_integral(1j * a, b, c, radius)
        yyz_val = utils.xxz_integral(1j * a, b, c, radius)
        zzz_val = utils.zzz_integral(a, b, c, radius)
        xzz_val = x_val - xxx_val - xyy_val
        yzz_val = y_val - xxy_val - yyy_val

        # degree 4

        xxxx_val = utils.xxxx_integral(a, b, c, radius)
        yyyy_val = utils.xxxx_integral(1j * a, b, c, radius)
        zzzz_val = utils.zzzz_integral(a, b, c, radius)
        xxzz_val = utils.xxzz_integral(a, b, c, radius)
        yyzz_val = utils.xxzz_integral(1j * a, b, c, radius)
        xxyy_val = xx_val - xxxx_val - xxzz_val

        xzzz_val = utils.xzzz_integral(a, b, c, radius)  #
        yzzz_val = -utils.xzzz_integral(1j * a, b, c, radius)  #
        xxxz_val = utils.xxxz_integral(a, b, c, radius)
        yyyz_val = -utils.xxxz_integral(1j * a, b, c, radius)  # Check this - possible error
        xyyz_val = xz_val - xxxz_val - xzzz_val
        xxyz_val = yz_val - yyyz_val - yzzz_val
        xxxy_val = utils.xxxy_integral(a, b, c, radius)
        xyyy_val = -utils.xxxy_integral(1j * a, b, c, radius)
        xyzz_val = xy_val - xxxy_val - xyyy_val

        # quartic quantities to compute the a_matrix restricted t the quadratic spherical harmonics

        q_11 = xxxx_val - 2 * xxyy_val + yyyy_val

        q_12 = xxxy_val - xyyy_val

        q_13 = xxxz_val - xyyz_val

        q_14 = xxyz_val - yyyz_val

        q_15 = 3 * (xxzz_val - yyzz_val) - (xx_val - yy_val)

        q_22 = xxyy_val
        q_23 = xxyz_val
        q_24 = xyyz_val
        q_25 = 3 * xyzz_val - xy_val

        q_33 = xxzz_val
        q_34 = xyzz_val
        q_35 = 3 * xzzz_val - xz_val

        q_44 = yyzz_val
        q_45 = 3 * yzzz_val - yz_val

        q_55 = 9 * zzzz_val - 6 * zz_val + v_val

        # .1
        a_matrix[0][0] += v_val
        a_matrix[0][1] += induced_1[0][0] * x_val + induced_1[1][0] * y_val + induced_1[2][0] * z_val
        a_matrix[0][2] += induced_1[0][1] * x_val + induced_1[1][1] * y_val + induced_1[2][1] * z_val
        a_matrix[0][3] += induced_1[0][2] * x_val + induced_1[1][2] * y_val + induced_1[2][2] * z_val
        a_matrix[0][4] += (induced_2[0][0] * (xx_val - yy_val) + induced_2[1][0] * xy_val + induced_2[2][0] * xz_val +
                           induced_2[3][0] * yz_val + induced_2[4][0] * (3 * zz_val - v_val))
        a_matrix[0][5] += (induced_2[0][1] * (xx_val - yy_val) + induced_2[1][1] * xy_val + induced_2[2][1] * xz_val +
                           induced_2[3][1] * yz_val + induced_2[4][1] * (3 * zz_val - v_val))
        a_matrix[0][6] += (induced_2[0][2] * (xx_val - yy_val) + induced_2[1][2] * xy_val + induced_2[2][2] * xz_val +
                           induced_2[3][2] * yz_val + induced_2[4][2] * (3 * zz_val - v_val))
        a_matrix[0][7] += (induced_2[0][3] * (xx_val - yy_val) + induced_2[1][3] * xy_val + induced_2[2][3] * xz_val +
                           induced_2[3][3] * yz_val + induced_2[4][3] * (3 * zz_val - v_val))
        a_matrix[0][8] += (induced_2[0][4] * (xx_val - yy_val) + induced_2[1][4] * xy_val + induced_2[2][4] * xz_val +
                           induced_2[3][4] * yz_val + induced_2[4][4] * (3 * zz_val - v_val))

        # X.
        a_matrix[1][1] += (induced_1[0][0] ** 2 * xx_val) + (2 * induced_1[0][0] * induced_1[1][0] * xy_val) + (
                    2 * induced_1[0][0] * induced_1[2][0] * xz_val) + (2 * induced_1[1][0] * induced_1[2][0] * yz_val) + (
                                      induced_1[1][0] * induced_1[1][0] * yy_val) + (
                                      induced_1[2][0] * induced_1[2][0] * zz_val)
        a_matrix[1][2] += (induced_1[0][0] * induced_1[0][1] * xx_val) + (
                    (induced_1[0][0] * induced_1[1][1] + induced_1[1][0] * induced_1[0][1]) * xy_val) + (
                                      (induced_1[0][0] * induced_1[2][1] + induced_1[2][0] * induced_1[0][1]) * xz_val) + (
                                      (induced_1[1][0] * induced_1[2][1] + induced_1[2][0] * induced_1[1][1]) * yz_val) + (
                                      induced_1[1][0] * induced_1[1][1] * yy_val) + (
                                      induced_1[2][0] * induced_1[2][1] * zz_val)
        a_matrix[1][3] += (induced_1[0][0] * induced_1[0][2] * xx_val) + (
                    (induced_1[0][0] * induced_1[1][2] + induced_1[1][0] * induced_1[0][2]) * xy_val) + (
                                      (induced_1[0][0] * induced_1[2][2] + induced_1[2][0] * induced_1[0][2]) * xz_val) + (
                                      (induced_1[1][0] * induced_1[2][2] + induced_1[2][0] * induced_1[1][2]) * yz_val) + (
                                      induced_1[1][0] * induced_1[1][2] * yy_val) + (
                                      induced_1[2][0] * induced_1[2][2] * zz_val)
        a_matrix[1][4] += induced_1[0][0] * induced_2[0][0] * (xxx_val - xyy_val) + induced_1[1][0] * induced_2[0][0] * (
                    xxy_val - yyy_val) + induced_1[2][0] * induced_2[0][0] * (xxz_val - yyz_val) + induced_1[0][0] * \
                          induced_2[1][0] * xxy_val + induced_1[1][0] * induced_2[1][0] * xyy_val + induced_1[2][0] * \
                          induced_2[1][0] * xyz_val + induced_1[0][0] * induced_2[2][0] * xxz_val + induced_1[1][0] * \
                          induced_2[2][0] * xyz_val + induced_1[2][0] * induced_2[2][0] * xzz_val + induced_1[0][0] * \
                          induced_2[3][0] * xyz_val + induced_1[1][0] * induced_2[3][0] * yyz_val + induced_1[2][0] * \
                          induced_2[3][0] * yzz_val + induced_1[0][0] * induced_2[4][0] * (3 * xzz_val - x_val) + \
                          induced_1[1][0] * induced_2[4][0] * (3 * yzz_val - y_val) + induced_1[2][0] * induced_2[4][0] * (
                                      3 * zzz_val - z_val)
        a_matrix[1][5] += (induced_1[0][0] * induced_2[0][1] * (xxx_val - xyy_val) + induced_1[1][0] * induced_2[0][1] * (
                    xxy_val - yyy_val) + induced_1[2][0] * induced_2[0][1] * (xxz_val - yyz_val) + induced_1[0][0] *
                           induced_2[1][1] * xxy_val + induced_1[1][0] * induced_2[1][1] * xyy_val + induced_1[2][0] *
                           induced_2[1][1] * xyz_val + induced_1[0][0] * induced_2[2][1] * xxz_val + induced_1[1][0] *
                           induced_2[2][1] * xyz_val + induced_1[2][0] * induced_2[2][1] * xzz_val + induced_1[0][0] *
                           induced_2[3][1] * xyz_val + induced_1[1][0] * induced_2[3][1] * yyz_val + induced_1[2][0] *
                           induced_2[3][1] * yzz_val + induced_1[0][0] * induced_2[4][1] * (3 * xzz_val - x_val) +
                           induced_1[1][0] * induced_2[4][1] * (3 * yzz_val - y_val) + induced_1[2][0] * induced_2[4][1] * (
                                       3 * zzz_val - z_val))
        a_matrix[1][6] += (induced_1[0][0] * induced_2[0][2] * (xxx_val - xyy_val) + induced_1[1][0] * induced_2[0][2] * (
                    xxy_val - yyy_val) + induced_1[2][0] * induced_2[0][2] * (xxz_val - yyz_val) + induced_1[0][0] *
                           induced_2[1][2] * xxy_val + induced_1[1][0] * induced_2[1][2] * xyy_val + induced_1[2][0] *
                           induced_2[1][2] * xyz_val + induced_1[0][0] * induced_2[2][2] * xxz_val + induced_1[1][0] *
                           induced_2[2][2] * xyz_val + induced_1[2][0] * induced_2[2][2] * xzz_val + induced_1[0][0] *
                           induced_2[3][2] * xyz_val + induced_1[1][0] * induced_2[3][2] * yyz_val + induced_1[2][0] *
                           induced_2[3][2] * yzz_val + induced_1[0][0] * induced_2[4][2] * (3 * xzz_val - x_val) +
                           induced_1[1][0] * induced_2[4][2] * (3 * yzz_val - y_val) + induced_1[2][0] * induced_2[4][2] * (
                                       3 * zzz_val - z_val))
        a_matrix[1][7] += (induced_1[0][0] * induced_2[0][3] * (xxx_val - xyy_val) + induced_1[1][0] * induced_2[0][3] * (
                    xxy_val - yyy_val) + induced_1[2][0] * induced_2[0][3] * (xxz_val - yyz_val) + induced_1[0][0] *
                           induced_2[1][3] * xxy_val + induced_1[1][0] * induced_2[1][3] * xyy_val + induced_1[2][0] *
                           induced_2[1][3] * xyz_val + induced_1[0][0] * induced_2[2][3] * xxz_val + induced_1[1][0] *
                           induced_2[2][3] * xyz_val + induced_1[2][0] * induced_2[2][3] * xzz_val + induced_1[0][0] *
                           induced_2[3][3] * xyz_val + induced_1[1][0] * induced_2[3][3] * yyz_val + induced_1[2][0] *
                           induced_2[3][3] * yzz_val + induced_1[0][0] * induced_2[4][3] * (3 * xzz_val - x_val) +
                           induced_1[1][0] * induced_2[4][3] * (3 * yzz_val - y_val) + induced_1[2][0] * induced_2[4][3] * (
                                       3 * zzz_val - z_val))
        a_matrix[1][8] += (induced_1[0][0] * induced_2[0][4] * (xxx_val - xyy_val) + induced_1[1][0] * induced_2[0][4] * (
                    xxy_val - yyy_val) + induced_1[2][0] * induced_2[0][4] * (xxz_val - yyz_val) + induced_1[0][0] *
                           induced_2[1][4] * xxy_val + induced_1[1][0] * induced_2[1][4] * xyy_val + induced_1[2][0] *
                           induced_2[1][4] * xyz_val + induced_1[0][0] * induced_2[2][4] * xxz_val + induced_1[1][0] *
                           induced_2[2][4] * xyz_val + induced_1[2][0] * induced_2[2][4] * xzz_val + induced_1[0][0] *
                           induced_2[3][4] * xyz_val + induced_1[1][0] * induced_2[3][4] * yyz_val + induced_1[2][0] *
                           induced_2[3][4] * yzz_val + induced_1[0][0] * induced_2[4][4] * (3 * xzz_val - x_val) +
                           induced_1[1][0] * induced_2[4][4] * (3 * yzz_val - y_val) + induced_1[2][0] * induced_2[4][4] * (
                                       3 * zzz_val - z_val))

        # .Y
        a_matrix[2][2] += (induced_1[0][1] ** 2 * xx_val) + (2 * induced_1[0][1] * induced_1[1][1] * xy_val) + (
                    2 * induced_1[0][1] * induced_1[2][1] * xz_val) + (2 * induced_1[1][1] * induced_1[2][1] * yz_val) + (
                                      induced_1[1][1] * induced_1[1][1] * yy_val) + (
                                      induced_1[2][1] * induced_1[2][1] * zz_val)
        a_matrix[2][3] += (induced_1[0][1] * induced_1[0][2] * xx_val) + (
                    (induced_1[0][1] * induced_1[1][2] + induced_1[1][1] * induced_1[0][2]) * xy_val) + (
                                      (induced_1[0][1] * induced_1[2][2] + induced_1[2][1] * induced_1[0][2]) * xz_val) + (
                                      (induced_1[1][1] * induced_1[2][2] + induced_1[2][1] * induced_1[1][2]) * yz_val) + (
                induced_1[1][1] * induced_1[1][2] * yy_val) + (
                                      induced_1[2][1] * induced_1[2][2] * zz_val)
        a_matrix[2][4] += (induced_1[0][1] * induced_2[0][0] * (xxx_val - xyy_val) + induced_1[1][1] * induced_2[0][0] * (
                    xxy_val - yyy_val) + induced_1[2][1] * induced_2[0][0] * (xxz_val - yyz_val) + induced_1[0][1] *
                           induced_2[1][0] * xxy_val + induced_1[1][1] * induced_2[1][0] * xyy_val + induced_1[2][1] *
                           induced_2[1][0] * xyz_val + induced_1[0][1] * induced_2[2][0] * xxz_val + induced_1[1][1] *
                           induced_2[2][0] * xyz_val + induced_1[2][1] * induced_2[2][0] * xzz_val + induced_1[0][1] *
                           induced_2[3][0] * xyz_val + induced_1[1][1] * induced_2[3][0] * yyz_val + induced_1[2][1] *
                           induced_2[3][0] * yzz_val + induced_1[0][1] * induced_2[4][0] * (3 * xzz_val - x_val) +
                           induced_1[1][1] * induced_2[4][0] * (3 * yzz_val - y_val) + induced_1[2][1] * induced_2[4][0] * (
                                       3 * zzz_val - z_val))
        a_matrix[2][5] += (induced_1[0][1] * induced_2[0][1] * (xxx_val - xyy_val) + induced_1[1][1] * induced_2[0][1] * (
                    xxy_val - yyy_val) + induced_1[2][1] * induced_2[0][1] * (xxz_val - yyz_val) + induced_1[0][1] *
                           induced_2[1][1] * xxy_val + induced_1[1][1] * induced_2[1][1] * xyy_val + induced_1[2][1] *
                           induced_2[1][1] * xyz_val + induced_1[0][1] * induced_2[2][1] * xxz_val + induced_1[1][1] *
                           induced_2[2][1] * xyz_val + induced_1[2][1] * induced_2[2][1] * xzz_val + induced_1[0][1] *
                           induced_2[3][1] * xyz_val + induced_1[1][1] * induced_2[3][1] * yyz_val + induced_1[2][1] *
                           induced_2[3][1] * yzz_val + induced_1[0][1] * induced_2[4][1] * (3 * xzz_val - x_val) +
                           induced_1[1][1] * induced_2[4][1] * (3 * yzz_val - y_val) + induced_1[2][1] * induced_2[4][1] * (
                                       3 * zzz_val - z_val))
        a_matrix[2][6] += (induced_1[0][1] * induced_2[0][2] * (xxx_val - xyy_val) + induced_1[1][1] * induced_2[0][2] * (
                    xxy_val - yyy_val) + induced_1[2][1] * induced_2[0][2] * (xxz_val - yyz_val) + induced_1[0][1] *
                           induced_2[1][2] * xxy_val + induced_1[1][1] * induced_2[1][2] * xyy_val + induced_1[2][1] *
                           induced_2[1][2] * xyz_val + induced_1[0][1] * induced_2[2][2] * xxz_val + induced_1[1][1] *
                           induced_2[2][2] * xyz_val + induced_1[2][1] * induced_2[2][2] * xzz_val + induced_1[0][1] *
                           induced_2[3][2] * xyz_val + induced_1[1][1] * induced_2[3][2] * yyz_val + induced_1[2][1] *
                           induced_2[3][2] * yzz_val + induced_1[0][1] * induced_2[4][2] * (3 * xzz_val - x_val) +
                           induced_1[1][1] * induced_2[4][2] * (3 * yzz_val - y_val) + induced_1[2][1] * induced_2[4][2] * (
                                       3 * zzz_val - z_val))
        a_matrix[2][7] += (induced_1[0][1] * induced_2[0][3] * (xxx_val - xyy_val) + induced_1[1][1] * induced_2[0][3] * (
                    xxy_val - yyy_val) + induced_1[2][1] * induced_2[0][3] * (xxz_val - yyz_val) + induced_1[0][1] *
                           induced_2[1][3] * xxy_val + induced_1[1][1] * induced_2[1][3] * xyy_val + induced_1[2][1] *
                           induced_2[1][3] * xyz_val + induced_1[0][1] * induced_2[2][3] * xxz_val + induced_1[1][1] *
                           induced_2[2][3] * xyz_val + induced_1[2][1] * induced_2[2][3] * xzz_val + induced_1[0][1] *
                           induced_2[3][3] * xyz_val + induced_1[1][1] * induced_2[3][3] * yyz_val + induced_1[2][1] *
                           induced_2[3][3] * yzz_val + induced_1[0][1] * induced_2[4][3] * (3 * xzz_val - x_val) +
                           induced_1[1][1] * induced_2[4][3] * (3 * yzz_val - y_val) + induced_1[2][1] * induced_2[4][3] * (
                                       3 * zzz_val - z_val))
        a_matrix[2][8] += (induced_1[0][1] * induced_2[0][4] * (xxx_val - xyy_val) + induced_1[1][1] * induced_2[0][4] * (
                    xxy_val - yyy_val) + induced_1[2][1] * induced_2[0][4] * (xxz_val - yyz_val) + induced_1[0][1] *
                           induced_2[1][4] * xxy_val + induced_1[1][1] * induced_2[1][4] * xyy_val + induced_1[2][1] *
                           induced_2[1][4] * xyz_val + induced_1[0][1] * induced_2[2][4] * xxz_val + induced_1[1][1] *
                           induced_2[2][4] * xyz_val + induced_1[2][1] * induced_2[2][4] * xzz_val + induced_1[0][1] *
                           induced_2[3][4] * xyz_val + induced_1[1][1] * induced_2[3][4] * yyz_val + induced_1[2][1] *
                           induced_2[3][4] * yzz_val + induced_1[0][1] * induced_2[4][4] * (3 * xzz_val - x_val) +
                           induced_1[1][1] * induced_2[4][4] * (3 * yzz_val - y_val) + induced_1[2][1] * induced_2[4][4] * (
                                       3 * zzz_val - z_val))

        # .Z
        a_matrix[3][3] += (induced_1[0][2] ** 2 * xx_val) + (2 * induced_1[0][2] * induced_1[1][2] * xy_val) + (
                    2 * induced_1[0][2] * induced_1[2][2] * xz_val) + (2 * induced_1[1][2] * induced_1[2][2] * yz_val) + (
                                      induced_1[1][2] * induced_1[1][2] * yy_val) + (
                                      induced_1[2][2] * induced_1[2][2] * zz_val)
        a_matrix[3][4] += (induced_1[0][2] * induced_2[0][0] * (xxx_val - xyy_val) + induced_1[1][2] * induced_2[0][0] * (
                    xxy_val - yyy_val) + induced_1[2][2] * induced_2[0][0] * (xxz_val - yyz_val) + induced_1[0][2] *
                           induced_2[1][0] * xxy_val + induced_1[1][2] * induced_2[1][0] * xyy_val + induced_1[2][2] *
                           induced_2[1][0] * xyz_val + induced_1[0][2] * induced_2[2][0] * xxz_val + induced_1[1][2] *
                           induced_2[2][0] * xyz_val + induced_1[2][2] * induced_2[2][0] * xzz_val + induced_1[0][2] *
                           induced_2[3][0] * xyz_val + induced_1[1][2] * induced_2[3][0] * yyz_val + induced_1[2][2] *
                           induced_2[3][0] * yzz_val + induced_1[0][2] * induced_2[4][0] * (3 * xzz_val - x_val) +
                           induced_1[1][2] * induced_2[4][0] * (3 * yzz_val - y_val) + induced_1[2][2] * induced_2[4][0] * (
                                       3 * zzz_val - z_val))
        a_matrix[3][5] += (induced_1[0][2] * induced_2[0][1] * (xxx_val - xyy_val) + induced_1[1][2] * induced_2[0][1] * (
                    xxy_val - yyy_val) + induced_1[2][2] * induced_2[0][1] * (xxz_val - yyz_val) + induced_1[0][2] *
                           induced_2[1][1] * xxy_val + induced_1[1][2] * induced_2[1][1] * xyy_val + induced_1[2][2] *
                           induced_2[1][1] * xyz_val + induced_1[0][2] * induced_2[2][1] * xxz_val + induced_1[1][2] *
                           induced_2[2][1] * xyz_val + induced_1[2][2] * induced_2[2][1] * xzz_val + induced_1[0][2] *
                           induced_2[3][1] * xyz_val + induced_1[1][2] * induced_2[3][1] * yyz_val + induced_1[2][2] *
                           induced_2[3][1] * yzz_val + induced_1[0][2] * induced_2[4][1] * (3 * xzz_val - x_val) +
                           induced_1[1][2] * induced_2[4][1] * (3 * yzz_val - y_val) + induced_1[2][2] * induced_2[4][1] * (
                                       3 * zzz_val - z_val))
        a_matrix[3][6] += (induced_1[0][2] * induced_2[0][2] * (xxx_val - xyy_val) + induced_1[1][2] * induced_2[0][2] * (
                    xxy_val - yyy_val) + induced_1[2][2] * induced_2[0][2] * (xxz_val - yyz_val) + induced_1[0][2] *
                           induced_2[1][2] * xxy_val + induced_1[1][2] * induced_2[1][2] * xyy_val + induced_1[2][2] *
                           induced_2[1][2] * xyz_val + induced_1[0][2] * induced_2[2][2] * xxz_val + induced_1[1][2] *
                           induced_2[2][2] * xyz_val + induced_1[2][2] * induced_2[2][2] * xzz_val + induced_1[0][2] *
                           induced_2[3][2] * xyz_val + induced_1[1][2] * induced_2[3][2] * yyz_val + induced_1[2][2] *
                           induced_2[3][2] * yzz_val + induced_1[0][2] * induced_2[4][2] * (3 * xzz_val - x_val) +
                           induced_1[1][2] * induced_2[4][2] * (3 * yzz_val - y_val) + induced_1[2][2] * induced_2[4][2] * (
                                       3 * zzz_val - z_val))
        a_matrix[3][7] += (induced_1[0][2] * induced_2[0][3] * (xxx_val - xyy_val) + induced_1[1][2] * induced_2[0][3] * (
                    xxy_val - yyy_val) + induced_1[2][2] * induced_2[0][3] * (xxz_val - yyz_val) + induced_1[0][2] *
                           induced_2[1][3] * xxy_val + induced_1[1][2] * induced_2[1][3] * xyy_val + induced_1[2][2] *
                           induced_2[1][3] * xyz_val + induced_1[0][2] * induced_2[2][3] * xxz_val + induced_1[1][2] *
                           induced_2[2][3] * xyz_val + induced_1[2][2] * induced_2[2][3] * xzz_val + induced_1[0][2] *
                           induced_2[3][3] * xyz_val + induced_1[1][2] * induced_2[3][3] * yyz_val + induced_1[2][2] *
                           induced_2[3][3] * yzz_val + induced_1[0][2] * induced_2[4][3] * (3 * xzz_val - x_val) +
                           induced_1[1][2] * induced_2[4][3] * (3 * yzz_val - y_val) + induced_1[2][2] * induced_2[4][3] * (
                                       3 * zzz_val - z_val))
        a_matrix[3][8] += (induced_1[0][2] * induced_2[0][4] * (xxx_val - xyy_val) + induced_1[1][2] * induced_2[0][4] * (
                    xxy_val - yyy_val) + induced_1[2][2] * induced_2[0][4] * (xxz_val - yyz_val) + induced_1[0][2] *
                           induced_2[1][4] * xxy_val + induced_1[1][2] * induced_2[1][4] * xyy_val + induced_1[2][2] *
                           induced_2[1][4] * xyz_val + induced_1[0][2] * induced_2[2][4] * xxz_val + induced_1[1][2] *
                           induced_2[2][4] * xyz_val + induced_1[2][2] * induced_2[2][4] * xzz_val + induced_1[0][2] *
                           induced_2[3][4] * xyz_val + induced_1[1][2] * induced_2[3][4] * yyz_val + induced_1[2][2] *
                           induced_2[3][4] * yzz_val + induced_1[0][2] * induced_2[4][4] * (3 * xzz_val - x_val) +
                           induced_1[1][2] * induced_2[4][4] * (3 * yzz_val - y_val) + induced_1[2][2] * induced_2[4][4] * (
                                       3 * zzz_val - z_val))

        # .(XX-YY)

        a_matrix[4][4] += (induced_2[0][0] * induced_2[0][0] * q_11) + (
                    (induced_2[0][0] * induced_2[1][0] + induced_2[1][0] * induced_2[0][0]) * q_12) + (
                                      (induced_2[0][0] * induced_2[2][0] + induced_2[2][0] * induced_2[0][0]) * q_13) + (
                                      (induced_2[0][0] * induced_2[3][0] + induced_2[3][0] * induced_2[0][0]) * q_14) + (
                                      (induced_2[0][0] * induced_2[4][0] + induced_2[4][0] * induced_2[0][0]) * q_15)
        a_matrix[4][4] += (induced_2[1][0] * induced_2[1][0] * q_22) + (
                (induced_2[1][0] * induced_2[2][0] + induced_2[2][0] * induced_2[1][0]) * q_23) + (
                                      (induced_2[1][0] * induced_2[3][0] + induced_2[3][0] * induced_2[1][0]) * q_24) + (
                                      (induced_2[1][0] * induced_2[4][0] + induced_2[4][0] * induced_2[1][0]) * q_25)
        a_matrix[4][4] += ((induced_2[2][0] * induced_2[2][0]) * q_33) + (
                    (induced_2[2][0] * induced_2[3][0] + induced_2[3][0] * induced_2[2][0]) * q_34) + (
                                      (induced_2[2][0] * induced_2[4][0] + induced_2[4][0] * induced_2[2][0]) * q_35)
        a_matrix[4][4] += ((induced_2[3][0] * induced_2[3][0]) * q_44) + (
                    (induced_2[3][0] * induced_2[4][0] + induced_2[4][0] * induced_2[3][0]) * q_45)
        a_matrix[4][4] += (induced_2[4][0] * induced_2[4][0] * q_55)

        a_matrix[4][5] += (induced_2[0][0] * induced_2[0][1] * q_11) + (
                    (induced_2[0][0] * induced_2[1][1] + induced_2[1][0] * induced_2[0][1]) * q_12) + (
                                      (induced_2[0][0] * induced_2[2][1] + induced_2[2][0] * induced_2[0][1]) * q_13) + (
                                      (induced_2[0][0] * induced_2[3][1] + induced_2[3][0] * induced_2[0][1]) * q_14) + (
                                      (induced_2[0][0] * induced_2[4][1] + induced_2[4][0] * induced_2[0][1]) * q_15)
        a_matrix[4][5] += (induced_2[1][0] * induced_2[1][1] * q_22) + (
                    (induced_2[1][0] * induced_2[2][1] + induced_2[2][0] * induced_2[1][1]) * q_23) + (
                                      (induced_2[1][0] * induced_2[3][1] + induced_2[3][0] * induced_2[1][1]) * q_24) + (
                                      (induced_2[1][0] * induced_2[4][1] + induced_2[4][0] * induced_2[1][1]) * q_25)
        a_matrix[4][5] += ((induced_2[2][0] * induced_2[2][1]) * q_33) + (
                    (induced_2[2][0] * induced_2[3][1] + induced_2[3][0] * induced_2[2][1]) * q_34) + (
                                      (induced_2[2][0] * induced_2[4][1] + induced_2[4][0] * induced_2[2][1]) * q_35)
        a_matrix[4][5] += ((induced_2[3][0] * induced_2[3][1]) * q_44) + (
                    (induced_2[3][0] * induced_2[4][1] + induced_2[4][0] * induced_2[3][1]) * q_45)
        a_matrix[4][5] += (induced_2[4][0] * induced_2[4][1] * q_55)

        a_matrix[4][6] += (induced_2[0][0] * induced_2[0][2] * q_11) + (
                    (induced_2[0][0] * induced_2[1][2] + induced_2[1][0] * induced_2[0][2]) * q_12) + (
                                      (induced_2[0][0] * induced_2[2][2] + induced_2[2][0] * induced_2[0][2]) * q_13) + (
                                      (induced_2[0][0] * induced_2[3][2] + induced_2[3][0] * induced_2[0][2]) * q_14) + (
                                      (induced_2[0][0] * induced_2[4][2] + induced_2[4][0] * induced_2[0][2]) * q_15)
        a_matrix[4][6] += (induced_2[1][0] * induced_2[1][2] * q_22) + (
                    (induced_2[1][0] * induced_2[2][2] + induced_2[2][0] * induced_2[1][2]) * q_23) + (
                                      (induced_2[1][0] * induced_2[3][2] + induced_2[3][0] * induced_2[1][2]) * q_24) + (
                                      (induced_2[1][0] * induced_2[4][2] + induced_2[4][0] * induced_2[1][2]) * q_25)
        a_matrix[4][6] += ((induced_2[2][0] * induced_2[2][2]) * q_33) + (
                    (induced_2[2][0] * induced_2[3][2] + induced_2[3][0] * induced_2[2][2]) * q_34) + (
                                      (induced_2[2][0] * induced_2[4][2] + induced_2[4][0] * induced_2[2][2]) * q_35)
        a_matrix[4][6] += ((induced_2[3][0] * induced_2[3][2]) * q_44) + (
                    (induced_2[3][0] * induced_2[4][2] + induced_2[4][0] * induced_2[3][2]) * q_45)
        a_matrix[4][6] += (induced_2[4][0] * induced_2[4][2] * q_55)

        a_matrix[4][7] += (induced_2[0][0] * induced_2[0][3] * q_11) + (
                    (induced_2[0][0] * induced_2[1][3] + induced_2[1][0] * induced_2[0][3]) * q_12) + (
                                      (induced_2[0][0] * induced_2[2][3] + induced_2[2][0] * induced_2[0][3]) * q_13) + (
                                      (induced_2[0][0] * induced_2[3][3] + induced_2[3][0] * induced_2[0][3]) * q_14) + (
                                      (induced_2[0][0] * induced_2[4][3] + induced_2[4][0] * induced_2[0][3]) * q_15)
        a_matrix[4][7] += (induced_2[1][0] * induced_2[1][3] * q_22) + (
                    (induced_2[1][0] * induced_2[2][3] + induced_2[2][0] * induced_2[1][3]) * q_23) + (
                                      (induced_2[1][0] * induced_2[3][3] + induced_2[3][0] * induced_2[1][3]) * q_24) + (
                                      (induced_2[1][0] * induced_2[4][3] + induced_2[4][0] * induced_2[1][3]) * q_25)
        a_matrix[4][7] += ((induced_2[2][0] * induced_2[2][3]) * q_33) + (
                    (induced_2[2][0] * induced_2[3][3] + induced_2[3][0] * induced_2[2][3]) * q_34) + (
                                      (induced_2[2][0] * induced_2[4][3] + induced_2[4][0] * induced_2[2][3]) * q_35)
        a_matrix[4][7] += ((induced_2[3][0] * induced_2[3][3]) * q_44) + (
                    (induced_2[3][0] * induced_2[4][3] + induced_2[4][0] * induced_2[3][3]) * q_45)
        a_matrix[4][7] += (induced_2[4][0] * induced_2[4][3] * q_55)

        a_matrix[4][8] += (induced_2[0][0] * induced_2[0][4] * q_11) + (
                    (induced_2[0][0] * induced_2[1][4] + induced_2[1][0] * induced_2[0][4]) * q_12) + (
                                      (induced_2[0][0] * induced_2[2][4] + induced_2[2][0] * induced_2[0][4]) * q_13) + (
                                      (induced_2[0][0] * induced_2[3][4] + induced_2[3][0] * induced_2[0][4]) * q_14) + (
                                      (induced_2[0][0] * induced_2[4][4] + induced_2[4][0] * induced_2[0][4]) * q_15)
        a_matrix[4][8] += (induced_2[1][0] * induced_2[1][4] * q_22) + (
                    (induced_2[1][0] * induced_2[2][4] + induced_2[2][0] * induced_2[1][4]) * q_23) + (
                                  (induced_2[1][0] * induced_2[3][4] + induced_2[3][0] * induced_2[1][4]) * q_24) + (
                                      (induced_2[1][0] * induced_2[4][4] + induced_2[4][0] * induced_2[1][4]) * q_25)
        a_matrix[4][8] += ((induced_2[2][0] * induced_2[2][4]) * q_33) + (
                    (induced_2[2][0] * induced_2[3][4] + induced_2[3][0] * induced_2[2][4]) * q_34) + (
                                      (induced_2[2][0] * induced_2[4][4] + induced_2[4][0] * induced_2[2][4]) * q_35)
        a_matrix[4][8] += ((induced_2[3][0] * induced_2[3][4]) * q_44) + (
                    (induced_2[3][0] * induced_2[4][4] + induced_2[4][0] * induced_2[3][4]) * q_45)
        a_matrix[4][8] += (induced_2[4][0] * induced_2[4][4] * q_55)

        # .XY
        a_matrix[5][5] += (induced_2[0][1] * induced_2[0][1] * q_11) + (
                    (induced_2[0][1] * induced_2[1][1] + induced_2[1][1] * induced_2[0][1]) * q_12) + (
                                      (induced_2[0][1] * induced_2[2][1] + induced_2[2][1] * induced_2[0][1]) * q_13) + (
                                      (induced_2[0][1] * induced_2[3][1] + induced_2[3][1] * induced_2[0][1]) * q_14) + (
                                      (induced_2[0][1] * induced_2[4][1] + induced_2[4][1] * induced_2[0][1]) * q_15)
        a_matrix[5][5] += (induced_2[1][1] * induced_2[1][1] * q_22) + (
                    (induced_2[1][1] * induced_2[2][1] + induced_2[2][1] * induced_2[1][1]) * q_23) + (
                                      (induced_2[1][1] * induced_2[3][1] + induced_2[3][1] * induced_2[1][1]) * q_24) + (
                                      (induced_2[1][1] * induced_2[4][1] + induced_2[4][1] * induced_2[1][1]) * q_25)
        a_matrix[5][5] += ((induced_2[2][1] * induced_2[2][1]) * q_33) + (
                    (induced_2[2][1] * induced_2[3][1] + induced_2[3][1] * induced_2[2][1]) * q_34) + (
                                      (induced_2[2][1] * induced_2[4][1] + induced_2[4][1] * induced_2[2][1]) * q_35)
        a_matrix[5][5] += ((induced_2[3][1] * induced_2[3][1]) * q_44) + (
                    (induced_2[3][1] * induced_2[4][1] + induced_2[4][1] * induced_2[3][1]) * q_45)
        a_matrix[5][5] += (induced_2[4][1] * induced_2[4][1] * q_55)

        a_matrix[5][6] += (induced_2[0][1] * induced_2[0][2] * q_11) + (
                    (induced_2[0][1] * induced_2[1][2] + induced_2[1][1] * induced_2[0][2]) * q_12) + (
                                      (induced_2[0][1] * induced_2[2][2] + induced_2[2][1] * induced_2[0][2]) * q_13) + (
                                      (induced_2[0][1] * induced_2[3][2] + induced_2[3][1] * induced_2[0][2]) * q_14) + (
                                      (induced_2[0][1] * induced_2[4][2] + induced_2[4][1] * induced_2[0][2]) * q_15)
        a_matrix[5][6] += (induced_2[1][1] * induced_2[1][2] * q_22) + (
                    (induced_2[1][1] * induced_2[2][2] + induced_2[2][1] * induced_2[1][2]) * q_23) + (
                                      (induced_2[1][1] * induced_2[3][2] + induced_2[3][1] * induced_2[1][2]) * q_24) + (
                                      (induced_2[1][1] * induced_2[4][2] + induced_2[4][1] * induced_2[1][2]) * q_25)
        a_matrix[5][6] += ((induced_2[2][1] * induced_2[2][2]) * q_33) + (
                    (induced_2[2][1] * induced_2[3][2] + induced_2[3][1] * induced_2[2][2]) * q_34) + (
                                      (induced_2[2][1] * induced_2[4][2] + induced_2[4][1] * induced_2[2][2]) * q_35)
        a_matrix[5][6] += ((induced_2[3][1] * induced_2[3][2]) * q_44) + (
                    (induced_2[3][1] * induced_2[4][2] + induced_2[4][1] * induced_2[3][2]) * q_45)
        a_matrix[5][6] += (induced_2[4][1] * induced_2[4][2] * q_55)

        a_matrix[5][7] += (induced_2[0][1] * induced_2[0][3] * q_11) + (
                    (induced_2[0][1] * induced_2[1][3] + induced_2[1][1] * induced_2[0][3]) * q_12) + (
                                      (induced_2[0][1] * induced_2[2][3] + induced_2[2][1] * induced_2[0][3]) * q_13) + (
                                      (induced_2[0][1] * induced_2[3][3] + induced_2[3][1] * induced_2[0][3]) * q_14) + (
                                      (induced_2[0][1] * induced_2[4][3] + induced_2[4][1] * induced_2[0][3]) * q_15)
        a_matrix[5][7] += (induced_2[1][1] * induced_2[1][3] * q_22) + (
                    (induced_2[1][1] * induced_2[2][3] + induced_2[2][1] * induced_2[1][3]) * q_23) + (
                                      (induced_2[1][1] * induced_2[3][3] + induced_2[3][1] * induced_2[1][3]) * q_24) + (
                (induced_2[1][1] * induced_2[4][3] + induced_2[4][1] * induced_2[1][3]) * q_25)
        a_matrix[5][7] += ((induced_2[2][1] * induced_2[2][3]) * q_33) + (
                    (induced_2[2][1] * induced_2[3][3] + induced_2[3][1] * induced_2[2][3]) * q_34) + (
                                      (induced_2[2][1] * induced_2[4][3] + induced_2[4][1] * induced_2[2][3]) * q_35)
        a_matrix[5][7] += ((induced_2[3][1] * induced_2[3][3]) * q_44) + (
                    (induced_2[3][1] * induced_2[4][3] + induced_2[4][1] * induced_2[3][3]) * q_45)
        a_matrix[5][7] += (induced_2[4][1] * induced_2[4][3] * q_55)

        a_matrix[5][8] += (induced_2[0][1] * induced_2[0][4] * q_11) + (
                    (induced_2[0][1] * induced_2[1][4] + induced_2[1][1] * induced_2[0][4]) * q_12) + (
                                      (induced_2[0][1] * induced_2[2][4] + induced_2[2][1] * induced_2[0][4]) * q_13) + (
                                      (induced_2[0][1] * induced_2[3][4] + induced_2[3][1] * induced_2[0][4]) * q_14) + (
                                      (induced_2[0][1] * induced_2[4][4] + induced_2[4][1] * induced_2[0][4]) * q_15)
        a_matrix[5][8] += (induced_2[1][1] * induced_2[1][4] * q_22) + (
                    (induced_2[1][1] * induced_2[2][4] + induced_2[2][1] * induced_2[1][4]) * q_23) + (
                                      (induced_2[1][1] * induced_2[3][4] + induced_2[3][1] * induced_2[1][4]) * q_24) + (
                                      (induced_2[1][1] * induced_2[4][4] + induced_2[4][1] * induced_2[1][4]) * q_25)
        a_matrix[5][8] += ((induced_2[2][1] * induced_2[2][4]) * q_33) + (
                    (induced_2[2][1] * induced_2[3][4] + induced_2[3][1] * induced_2[2][4]) * q_34) + (
                                      (induced_2[2][1] * induced_2[4][4] + induced_2[4][1] * induced_2[2][4]) * q_35)
        a_matrix[5][8] += ((induced_2[3][1] * induced_2[3][4]) * q_44) + (
                    (induced_2[3][1] * induced_2[4][4] + induced_2[4][1] * induced_2[3][4]) * q_45)
        a_matrix[5][8] += (induced_2[4][1] * induced_2[4][4] * q_55)

        # .XZ
        a_matrix[6][6] += (induced_2[0][2] * induced_2[0][2] * q_11) + (
                    (induced_2[0][2] * induced_2[1][2] + induced_2[1][2] * induced_2[0][2]) * q_12) + (
                                      (induced_2[0][2] * induced_2[2][2] + induced_2[2][2] * induced_2[0][2]) * q_13) + (
                                      (induced_2[0][2] * induced_2[3][2] + induced_2[3][2] * induced_2[0][2]) * q_14) + (
                                      (induced_2[0][2] * induced_2[4][2] + induced_2[4][2] * induced_2[0][2]) * q_15)
        a_matrix[6][6] += (induced_2[1][2] * induced_2[1][2] * q_22) + (
                    (induced_2[1][2] * induced_2[2][2] + induced_2[2][2] * induced_2[1][2]) * q_23) + (
                                      (induced_2[1][2] * induced_2[3][2] + induced_2[3][2] * induced_2[1][2]) * q_24) + (
                                      (induced_2[1][2] * induced_2[4][2] + induced_2[4][2] * induced_2[1][2]) * q_25)
        a_matrix[6][6] += ((induced_2[2][2] * induced_2[2][2]) * q_33) + (
                    (induced_2[2][2] * induced_2[3][2] + induced_2[3][2] * induced_2[2][2]) * q_34) + (
                                      (induced_2[2][2] * induced_2[4][2] + induced_2[4][2] * induced_2[2][2]) * q_35)
        a_matrix[6][6] += ((induced_2[3][2] * induced_2[3][2]) * q_44) + (
                    (induced_2[3][2] * induced_2[4][2] + induced_2[4][2] * induced_2[3][2]) * q_45)
        a_matrix[6][6] += (induced_2[4][2] * induced_2[4][2] * q_55)

        a_matrix[6][7] += (induced_2[0][2] * induced_2[0][3] * q_11) + (
                    (induced_2[0][2] * induced_2[1][3] + induced_2[1][2] * induced_2[0][3]) * q_12) + (
                                      (induced_2[0][2] * induced_2[2][3] + induced_2[2][2] * induced_2[0][3]) * q_13) + (
                                      (induced_2[0][2] * induced_2[3][3] + induced_2[3][2] * induced_2[0][3]) * q_14) + (
                                      (induced_2[0][2] * induced_2[4][3] + induced_2[4][2] * induced_2[0][3]) * q_15)
        a_matrix[6][7] += (induced_2[1][2] * induced_2[1][3] * q_22) + (
                    (induced_2[1][2] * induced_2[2][3] + induced_2[2][2] * induced_2[1][3]) * q_23) + (
                                      (induced_2[1][2] * induced_2[3][3] + induced_2[3][2] * induced_2[1][3]) * q_24) + (
                                      (induced_2[1][2] * induced_2[4][3] + induced_2[4][2] * induced_2[1][3]) * q_25)
        a_matrix[6][7] += ((induced_2[2][2] * induced_2[2][3]) * q_33) + (
                    (induced_2[2][2] * induced_2[3][3] + induced_2[3][2] * induced_2[2][3]) * q_34) + (
                                      (induced_2[2][2] * induced_2[4][3] + induced_2[4][2] * induced_2[2][3]) * q_35)
        a_matrix[6][7] += ((induced_2[3][2] * induced_2[3][3]) * q_44) + (
                    (induced_2[3][2] * induced_2[4][3] + induced_2[4][2] * induced_2[3][3]) * q_45)
        a_matrix[6][7] += (induced_2[4][2] * induced_2[4][3] * q_55)

        a_matrix[6][8] += (induced_2[0][2] * induced_2[0][4] * q_11) + (
                    (induced_2[0][2] * induced_2[1][4] + induced_2[1][2] * induced_2[0][4]) * q_12) + (
                                      (induced_2[0][2] * induced_2[2][4] + induced_2[2][2] * induced_2[0][4]) * q_13) + (
                                      (induced_2[0][2] * induced_2[3][4] + induced_2[3][2] * induced_2[0][4]) * q_14) + (
                                      (induced_2[0][2] * induced_2[4][4] + induced_2[4][2] * induced_2[0][4]) * q_15)
        a_matrix[6][8] += (induced_2[1][2] * induced_2[1][4] * q_22) + (
                    (induced_2[1][2] * induced_2[2][4] + induced_2[2][2] * induced_2[1][4]) * q_23) + (
                                      (induced_2[1][2] * induced_2[3][4] + induced_2[3][2] * induced_2[1][4]) * q_24) + (
                                      (induced_2[1][2] * induced_2[4][4] + induced_2[4][2] * induced_2[1][4]) * q_25)
        a_matrix[6][8] += ((induced_2[2][2] * induced_2[2][4]) * q_33) + (
                    (induced_2[2][2] * induced_2[3][4] + induced_2[3][2] * induced_2[2][4]) * q_34) + (
                                      (induced_2[2][2] * induced_2[4][4] + induced_2[4][2] * induced_2[2][4]) * q_35)
        a_matrix[6][8] += ((induced_2[3][2] * induced_2[3][4]) * q_44) + (
                    (induced_2[3][2] * induced_2[4][4] + induced_2[4][2] * induced_2[3][4]) * q_45)
        a_matrix[6][8] += (induced_2[4][2] * induced_2[4][4] * q_55)

        # .YZ
        a_matrix[7][7] += (induced_2[0][3] * induced_2[0][3] * q_11) + (
                (induced_2[0][3] * induced_2[1][3] + induced_2[1][3] * induced_2[0][3]) * q_12) + (
                                  (induced_2[0][3] * induced_2[2][3] + induced_2[2][3] * induced_2[0][3]) * q_13) + (
                                  (induced_2[0][3] * induced_2[3][3] + induced_2[3][3] * induced_2[0][3]) * q_14) + (
                                      (induced_2[0][3] * induced_2[4][3] + induced_2[4][3] * induced_2[0][3]) * q_15)
        a_matrix[7][7] += (induced_2[1][3] * induced_2[1][3] * q_22) + (
                (induced_2[1][3] * induced_2[2][3] + induced_2[2][3] * induced_2[1][3]) * q_23) + (
                                      (induced_2[1][3] * induced_2[3][3] + induced_2[3][3] * induced_2[1][3]) * q_24) + (
                                      (induced_2[1][3] * induced_2[4][3] + induced_2[4][3] * induced_2[1][3]) * q_25)
        a_matrix[7][7] += ((induced_2[2][3] * induced_2[2][3]) * q_33) + (
                    (induced_2[2][3] * induced_2[3][3] + induced_2[3][3] * induced_2[2][3]) * q_34) + (
                                  (induced_2[2][3] * induced_2[4][3] + induced_2[4][3] * induced_2[2][3]) * q_35)
        a_matrix[7][7] += ((induced_2[3][3] * induced_2[3][3]) * q_44) + (
                    (induced_2[3][3] * induced_2[4][3] + induced_2[4][3] * induced_2[3][3]) * q_45)
        a_matrix[7][7] += (induced_2[4][3] * induced_2[4][3] * q_55)

        a_matrix[7][8] += (induced_2[0][3] * induced_2[0][4] * q_11) + (
                (induced_2[0][3] * induced_2[1][4] + induced_2[1][3] * induced_2[0][4]) * q_12) + (
                                  (induced_2[0][3] * induced_2[2][4] + induced_2[2][3] * induced_2[0][4]) * q_13) + (
                                      (induced_2[0][3] * induced_2[3][4] + induced_2[3][3] * induced_2[0][4]) * q_14) + (
                                      (induced_2[0][3] * induced_2[4][4] + induced_2[4][3] * induced_2[0][4]) * q_15)
        a_matrix[7][8] += (induced_2[1][3] * induced_2[1][4] * q_22) + (
                    (induced_2[1][3] * induced_2[2][4] + induced_2[2][3] * induced_2[1][4]) * q_23) + (
                                      (induced_2[1][3] * induced_2[3][4] + induced_2[3][3] * induced_2[1][4]) * q_24) + (
                                      (induced_2[1][3] * induced_2[4][4] + induced_2[4][3] * induced_2[1][4]) * q_25)
        a_matrix[7][8] += ((induced_2[2][3] * induced_2[2][4]) * q_33) + (
                    (induced_2[2][3] * induced_2[3][4] + induced_2[3][3] * induced_2[2][4]) * q_34) + (
                                      (induced_2[2][3] * induced_2[4][4] + induced_2[4][3] * induced_2[2][4]) * q_35)
        a_matrix[7][8] += ((induced_2[3][3] * induced_2[3][4]) * q_44) + (
                    (induced_2[3][3] * induced_2[4][4] + induced_2[4][3] * induced_2[3][4]) * q_45)
        a_matrix[7][8] += (induced_2[4][3] * induced_2[4][4] * q_55)

        a_matrix[8][8] += (induced_2[0][4] * induced_2[0][4] * q_11) + (
                    (induced_2[0][4] * induced_2[1][4] + induced_2[1][4] * induced_2[0][4]) * q_12) + (
                                      (induced_2[0][4] * induced_2[2][4] + induced_2[2][4] * induced_2[0][4]) * q_13) + (
                                      (induced_2[0][4] * induced_2[3][4] + induced_2[3][4] * induced_2[0][4]) * q_14) + (
                                      (induced_2[0][4] * induced_2[4][4] + induced_2[4][4] * induced_2[0][4]) * q_15)
        a_matrix[8][8] += (induced_2[1][4] * induced_2[1][4] * q_22) + (
                    (induced_2[1][4] * induced_2[2][4] + induced_2[2][4] * induced_2[1][4]) * q_23) + (
                                      (induced_2[1][4] * induced_2[3][4] + induced_2[3][4] * induced_2[1][4]) * q_24) + (
                                      (induced_2[1][4] * induced_2[4][4] + induced_2[4][4] * induced_2[1][4]) * q_25)
        a_matrix[8][8] += ((induced_2[2][4] * induced_2[2][4]) * q_33) + (
                    (induced_2[2][4] * induced_2[3][4] + induced_2[3][4] * induced_2[2][4]) * q_34) + (
                                      (induced_2[2][4] * induced_2[4][4] + induced_2[4][4] * induced_2[2][4]) * q_35)
        a_matrix[8][8] += ((induced_2[3][4] * induced_2[3][4]) * q_44) + (
                    (induced_2[3][4] * induced_2[4][4] + induced_2[4][4] * induced_2[3][4]) * q_45)
        a_matrix[8][8] += (induced_2[4][4] * induced_2[4][4] * q_55)

        # Intersecting spheres

        for i_up in range(0, len(next_level[sphere])):
            s_up = next_level[sphere][i_up]

            radius = transformed_radii[s_up]
            d_inv = transformations[s_up]

            a = d_1_mat_t[level][sphere]
            b = d_2_mat_t[level][sphere]
            c = d_0_mat_t[level][sphere]

            alpha = d_inv[0][0]
            beta = d_inv[0][1]

            k_2 = abs(alpha + (a * np.conj(beta))) ** 2 + (b * abs(beta) ** 2)
            k_1 = ((alpha + a * np.conj(beta)) * np.conj(beta - a * np.conj(alpha))) - b * alpha * np.conj(beta)
            k_0 = abs(beta - a * np.conj(alpha)) ** 2 + b * abs(alpha) ** 2

            k_1 = k_1 / k_2
            k_0 = k_0 / k_2
            c = c / (k_2 ** 2)

            a = -np.conj(k_1)
            b = k_0 - abs(a) ** 2

            induced_1 = np.array([
                [(alpha ** 2 - np.conj(beta) ** 2).real, (alpha ** 2 + np.conj(beta) ** 2).imag,
                 -(2 * alpha * np.conj(beta)).real],
                [-(alpha ** 2 - np.conj(beta) ** 2).imag, (alpha ** 2 + np.conj(beta) ** 2).real,
                 (2 * alpha * np.conj(beta)).imag],
                [alpha * beta + np.conj(alpha * beta), 2 * (alpha * beta).imag, abs(alpha) ** 2 - abs(beta) ** 2]
            ])

            b_r = beta.real
            b_i = beta.imag

            induced_2 = np.array([
                [alpha ** 4 + b_i ** 4 + b_r ** 4 - 6 * b_i ** 2 * b_r ** 2, -2 * b_i * b_r * (b_i ** 2 - b_r ** 2),
                 -alpha * b_r * (alpha ** 2 + 3 * b_i ** 2 - b_r ** 2),
                 alpha * b_i * (alpha ** 2 - b_i ** 2 + 3 * b_r ** 2), 6 * alpha ** 2 * (b_r ** 2 - b_i ** 2)],
                [-8 * b_i * b_r * (b_i ** 2 - b_r ** 2), alpha ** 4 - b_i ** 4 - b_r ** 4 + 6 * b_i ** 2 * b_r ** 2,
                 -2 * alpha * b_i * (alpha ** 2 + b_i ** 2 - 3 * b_r ** 2),
                 -2 * alpha * b_r * (alpha ** 2 - 3 * b_i ** 2 + b_r ** 2), 24 * alpha ** 2 * b_i * b_r],
                [4 * alpha * b_r * (alpha ** 2 + 3 * b_i ** 2 - b_r ** 2),
                 2 * alpha * b_i * (alpha ** 2 - 3 * b_r ** 2 + b_i ** 2),
                 alpha ** 4 - 6 * alpha ** 2 * b_r ** 2 - b_i ** 4 + b_r ** 4,
                 2 * b_i * b_r * (b_i ** 2 + b_r ** 2 - 3 * alpha ** 2),
                 12 * alpha * b_r * (b_i ** 2 + b_r ** 2 - alpha ** 2)],
                [-4 * alpha * b_i * (alpha ** 2 - b_i ** 2 + 3 * b_r ** 2),
                 2 * alpha * b_r * (alpha ** 2 + b_r ** 2 - 3 * b_i ** 2),
                 2 * b_i * b_r * (b_i ** 2 + b_r ** 2 - 3 * alpha ** 2),
                 alpha ** 4 + b_i ** 4 - b_r ** 4 - 6 * alpha ** 2 * b_i ** 2,
                 12 * alpha * b_i * (b_i ** 2 + b_r ** 2 - alpha ** 2)],
                [-2 * alpha ** 2 * (b_i ** 2 - b_r ** 2), 2 * alpha ** 2 * b_i * b_r,
                 -alpha * b_r * (b_i ** 2 + b_r ** 2 - alpha ** 2), -alpha * b_i * (b_i ** 2 + b_r ** 2 - alpha ** 2),
                 6 * b_i ** 4 + 12 * b_i ** 2 * b_r ** 2 + 6 * b_r ** 4 - 6 * (b_i ** 2 + b_r ** 2) + 1]
            ])

            # degree 0
            v_val = utils.vol_integral(a, b, c, radius)

            # degree 1

            x_val = utils.x_integral(a, b, c, radius)
            y_val = -utils.x_integral(1j * a, b, c, radius)
            z_val = utils.z_integral(a, b, c, radius)

            # degree 2

            xx_val = utils.xx_integral(a, b, c, radius)
            yy_val = utils.xx_integral(1j * a, b, c, radius)
            zz_val = v_val - xx_val - yy_val
            xy_val = utils.xy_integral(a, b, c, radius)
            xz_val = utils.xz_integral(a, b, c, radius)
            yz_val = -utils.xz_integral(1j * a, b, c, radius)

            # degree 3

            xxx_val = utils.xxx_integral(a, b, c, radius)
            xxy_val = utils.xxy_integral(a, b, c, radius)
            xxz_val = utils.xxz_integral(a, b, c, radius)
            xyy_val = utils.xxy_integral(1j * a, b, c, radius)
            xyz_val = utils.xyz_integral(a, b, c, radius)
            yyy_val = -utils.xxx_integral(1j * a, b, c, radius)
            yyz_val = utils.xxz_integral(1j * a, b, c, radius)
            zzz_val = utils.zzz_integral(a, b, c, radius)
            xzz_val = x_val - xxx_val - xyy_val
            yzz_val = y_val - xxy_val - yyy_val

            # degree 4

            xxxx_val = utils.xxxx_integral(a, b, c, radius)
            yyyy_val = utils.xxxx_integral(1j * a, b, c, radius)
            zzzz_val = utils.zzzz_integral(a, b, c, radius)
            xxzz_val = utils.xxzz_integral(a, b, c, radius)
            yyzz_val = utils.xxzz_integral(1j * a, b, c, radius)
            xxyy_val = xx_val - xxxx_val - xxzz_val

            xzzz_val = utils.xzzz_integral(a, b, c, radius)  #
            yzzz_val = -utils.xzzz_integral(1j * a, b, c, radius)  #
            xxxz_val = utils.xxxz_integral(a, b, c, radius)
            yyyz_val = -utils.xxxz_integral(1j * a, b, c, radius)  # Check this - possible error
            xyyz_val = xz_val - xxxz_val - xzzz_val
            xxyz_val = yz_val - yyyz_val - yzzz_val
            xxxy_val = utils.xxxy_integral(a, b, c, radius)
            xyyy_val = -utils.xxxy_integral(1j * a, b, c, radius)
            xyzz_val = xy_val - xxxy_val - xyyy_val

            # quartic quantities to compute the a_matrix restricted t the quadratic spherical harmonics

            q_11 = xxxx_val - 2 * xxyy_val + yyyy_val
            q_12 = xxxy_val - xyyy_val
            q_13 = xxxz_val - xyyz_val
            q_14 = xxyz_val - yyyz_val
            q_15 = 3 * (xxzz_val - yyzz_val) - (xx_val - yy_val)

            q_22 = xxyy_val
            q_23 = xxyz_val
            q_24 = xyyz_val
            q_25 = 3 * xyzz_val - xy_val

            q_33 = xxzz_val
            q_34 = xyzz_val
            q_35 = 3 * xzzz_val - xz_val

            q_44 = yyzz_val
            q_45 = 3 * yzzz_val - yz_val

            q_55 = 9 * zzzz_val - 6 * zz_val + v_val

            a_matrix[0][0] += -v_val
            a_matrix[0][1] += -induced_1[0][0] * x_val - induced_1[1][0] * y_val - induced_1[2][0] * z_val
            a_matrix[0][2] += -induced_1[0][1] * x_val - induced_1[1][1] * y_val - induced_1[2][1] * z_val
            a_matrix[0][3] += -induced_1[0][2] * x_val - induced_1[1][2] * y_val - induced_1[2][2] * z_val
            a_matrix[0][4] += -(induced_2[0][0] * (xx_val - yy_val) + induced_2[1][0] * xy_val + induced_2[2][0] * xz_val +
                                induced_2[3][0] * yz_val + induced_2[4][0] * (3 * zz_val - v_val))
            a_matrix[0][5] += -(induced_2[0][1] * (xx_val - yy_val) + induced_2[1][1] * xy_val + induced_2[2][1] * xz_val +
                                induced_2[3][1] * yz_val + induced_2[4][1] * (3 * zz_val - v_val))
            a_matrix[0][6] += -(induced_2[0][2] * (xx_val - yy_val) + induced_2[1][2] * xy_val + induced_2[2][2] * xz_val +
                                induced_2[3][2] * yz_val + induced_2[4][2] * (3 * zz_val - v_val))
            a_matrix[0][7] += -(induced_2[0][3] * (xx_val - yy_val) + induced_2[1][3] * xy_val + induced_2[2][3] * xz_val +
                                induced_2[3][3] * yz_val + induced_2[4][3] * (3 * zz_val - v_val))
            a_matrix[0][8] += -(induced_2[0][4] * (xx_val - yy_val) + induced_2[1][4] * xy_val + induced_2[2][4] * xz_val +
                                induced_2[3][4] * yz_val + induced_2[4][4] * (3 * zz_val - v_val))

            a_matrix[1][1] += -((induced_1[0][0] ** 2 * xx_val) + (2 * induced_1[0][0] * induced_1[1][0] * xy_val) + (
                        2 * induced_1[0][0] * induced_1[2][0] * xz_val) + (
                                            2 * induced_1[1][0] * induced_1[2][0] * yz_val) + (
                                            induced_1[1][0] * induced_1[1][0] * yy_val) + (
                                            induced_1[2][0] * induced_1[2][0] * zz_val))
            a_matrix[1][2] += -((induced_1[0][0] * induced_1[0][1] * xx_val) + (
                        (induced_1[0][0] * induced_1[1][1] + induced_1[1][0] * induced_1[0][1]) * xy_val) + ((induced_1[0][
                                                                                                                  0] *
                                                                                                              induced_1[2][
                                                                                                                  1] +
                                                                                                              induced_1[2][
                                                                                                                  0] *
                                                                                                              induced_1[0][
                                                                                                                  1]) * xz_val) + (
                                            (induced_1[1][0] * induced_1[2][1] + induced_1[2][0] * induced_1[1][
                                                1]) * yz_val) + (induced_1[1][0] * induced_1[1][1] * yy_val) + (
                                            induced_1[2][0] * induced_1[2][1] * zz_val))
            a_matrix[1][3] += -((induced_1[0][0] * induced_1[0][2] * xx_val) + (
                        (induced_1[0][0] * induced_1[1][2] + induced_1[1][0] * induced_1[0][2]) * xy_val) + ((induced_1[0][
                                                                                                                  0] *
                                                                                                              induced_1[2][
                                                                                                                  2] +
                                                                                                              induced_1[2][
                                                                                                                  0] *
                                                                                                              induced_1[0][
                                                                                                                  2]) * xz_val) + (
                                            (induced_1[1][0] * induced_1[2][2] + induced_1[2][0] * induced_1[1][
                                                2]) * yz_val) + (induced_1[1][0] * induced_1[1][2] * yy_val) + (
                                            induced_1[2][0] * induced_1[2][2] * zz_val))
            a_matrix[1][4] += -(
                        induced_1[0][0] * induced_2[0][0] * (xxx_val - xyy_val) + induced_1[1][0] * induced_2[0][0] * (
                            xxy_val - yyy_val) + induced_1[2][0] * induced_2[0][0] * (xxz_val - yyz_val) + induced_1[0][0] *
                        induced_2[1][0] * xxy_val + induced_1[1][0] * induced_2[1][0] * xyy_val + induced_1[2][0] *
                        induced_2[1][0] * xyz_val + induced_1[0][0] * induced_2[2][0] * xxz_val + induced_1[1][0] *
                        induced_2[2][0] * xyz_val + induced_1[2][0] * induced_2[2][0] * xzz_val + induced_1[0][0] *
                        induced_2[3][0] * xyz_val + induced_1[1][0] * induced_2[3][0] * yyz_val + induced_1[2][0] *
                        induced_2[3][0] * yzz_val + induced_1[0][0] * induced_2[4][0] * (3 * xzz_val - x_val) +
                        induced_1[1][0] * induced_2[4][0] * (3 * yzz_val - y_val) + induced_1[2][0] * induced_2[4][0] * (
                                    3 * zzz_val - z_val))
            a_matrix[1][5] += -(
                        induced_1[0][0] * induced_2[0][1] * (xxx_val - xyy_val) + induced_1[1][0] * induced_2[0][1] * (
                            xxy_val - yyy_val) + induced_1[2][0] * induced_2[0][1] * (xxz_val - yyz_val) + induced_1[0][0] *
                        induced_2[1][1] * xxy_val + induced_1[1][0] * induced_2[1][1] * xyy_val + induced_1[2][0] *
                        induced_2[1][1] * xyz_val + induced_1[0][0] * induced_2[2][1] * xxz_val + induced_1[1][0] *
                        induced_2[2][1] * xyz_val + induced_1[2][0] * induced_2[2][1] * xzz_val + induced_1[0][0] *
                        induced_2[3][1] * xyz_val + induced_1[1][0] * induced_2[3][1] * yyz_val + induced_1[2][0] *
                        induced_2[3][1] * yzz_val + induced_1[0][0] * induced_2[4][1] * (3 * xzz_val - x_val) +
                        induced_1[1][0] * induced_2[4][1] * (3 * yzz_val - y_val) + induced_1[2][0] * induced_2[4][1] * (
                                    3 * zzz_val - z_val))
            a_matrix[1][6] += -(
                        induced_1[0][0] * induced_2[0][2] * (xxx_val - xyy_val) + induced_1[1][0] * induced_2[0][2] * (
                            xxy_val - yyy_val) + induced_1[2][0] * induced_2[0][2] * (xxz_val - yyz_val) + induced_1[0][0] *
                        induced_2[1][2] * xxy_val + induced_1[1][0] * induced_2[1][2] * xyy_val + induced_1[2][0] *
                        induced_2[1][2] * xyz_val + induced_1[0][0] * induced_2[2][2] * xxz_val + induced_1[1][0] *
                        induced_2[2][2] * xyz_val + induced_1[2][0] * induced_2[2][2] * xzz_val + induced_1[0][0] *
                        induced_2[3][2] * xyz_val + induced_1[1][0] * induced_2[3][2] * yyz_val + induced_1[2][0] *
                        induced_2[3][2] * yzz_val + induced_1[0][0] * induced_2[4][2] * (3 * xzz_val - x_val) +
                        induced_1[1][0] * induced_2[4][2] * (3 * yzz_val - y_val) + induced_1[2][0] * induced_2[4][2] * (
                                    3 * zzz_val - z_val))
            a_matrix[1][7] += -(
                        induced_1[0][0] * induced_2[0][3] * (xxx_val - xyy_val) + induced_1[1][0] * induced_2[0][3] * (
                            xxy_val - yyy_val) + induced_1[2][0] * induced_2[0][3] * (xxz_val - yyz_val) + induced_1[0][0] *
                        induced_2[1][3] * xxy_val + induced_1[1][0] * induced_2[1][3] * xyy_val + induced_1[2][0] *
                        induced_2[1][3] * xyz_val + induced_1[0][0] * induced_2[2][3] * xxz_val + induced_1[1][0] *
                        induced_2[2][3] * xyz_val + induced_1[2][0] * induced_2[2][3] * xzz_val + induced_1[0][0] *
                        induced_2[3][3] * xyz_val + induced_1[1][0] * induced_2[3][3] * yyz_val + induced_1[2][0] *
                        induced_2[3][3] * yzz_val + induced_1[0][0] * induced_2[4][3] * (3 * xzz_val - x_val) +
                        induced_1[1][0] * induced_2[4][3] * (3 * yzz_val - y_val) + induced_1[2][0] * induced_2[4][3] * (
                                    3 * zzz_val - z_val))
            a_matrix[1][8] += -(
                        induced_1[0][0] * induced_2[0][4] * (xxx_val - xyy_val) + induced_1[1][0] * induced_2[0][4] * (
                            xxy_val - yyy_val) + induced_1[2][0] * induced_2[0][4] * (xxz_val - yyz_val) + induced_1[0][0] *
                        induced_2[1][4] * xxy_val + induced_1[1][0] * induced_2[1][4] * xyy_val + induced_1[2][0] *
                        induced_2[1][4] * xyz_val + induced_1[0][0] * induced_2[2][4] * xxz_val + induced_1[1][0] *
                        induced_2[2][4] * xyz_val + induced_1[2][0] * induced_2[2][4] * xzz_val + induced_1[0][0] *
                        induced_2[3][4] * xyz_val + induced_1[1][0] * induced_2[3][4] * yyz_val + induced_1[2][0] *
                        induced_2[3][4] * yzz_val + induced_1[0][0] * induced_2[4][4] * (3 * xzz_val - x_val) +
                        induced_1[1][0] * induced_2[4][4] * (3 * yzz_val - y_val) + induced_1[2][0] * induced_2[4][4] * (
                                    3 * zzz_val - z_val))

            a_matrix[2][2] += -((induced_1[0][1] ** 2 * xx_val) + (2 * induced_1[0][1] * induced_1[1][1] * xy_val) + (
                        2 * induced_1[0][1] * induced_1[2][1] * xz_val) + (
                                            2 * induced_1[1][1] * induced_1[2][1] * yz_val) + (
                                            induced_1[1][1] * induced_1[1][1] * yy_val) + (
                                            induced_1[2][1] * induced_1[2][1] * zz_val))
            a_matrix[2][3] += -((induced_1[0][1] * induced_1[0][2] * xx_val) + (
                        (induced_1[0][1] * induced_1[1][2] + induced_1[1][1] * induced_1[0][2]) * xy_val) + ((induced_1[0][
                                                                                                                  1] *
                                                                                                              induced_1[2][
                                                                                                                  2] +
                                                                                                              induced_1[2][
                                                                                                                  1] *
                                                                                                              induced_1[0][
                                                                                                                  2]) * xz_val) + (
                                            (induced_1[1][1] * induced_1[2][2] + induced_1[2][1] * induced_1[1][
                                                2]) * yz_val) + (induced_1[1][1] * induced_1[1][2] * yy_val) + (
                                            induced_1[2][1] * induced_1[2][2] * zz_val))
            a_matrix[2][4] += -(
                        induced_1[0][1] * induced_2[0][0] * (xxx_val - xyy_val) + induced_1[1][1] * induced_2[0][0] * (
                            xxy_val - yyy_val) + induced_1[2][1] * induced_2[0][0] * (xxz_val - yyz_val) + induced_1[0][1] *
                        induced_2[1][0] * xxy_val + induced_1[1][1] * induced_2[1][0] * xyy_val + induced_1[2][1] *
                        induced_2[1][0] * xyz_val + induced_1[0][1] * induced_2[2][0] * xxz_val + induced_1[1][1] *
                        induced_2[2][0] * xyz_val + induced_1[2][1] * induced_2[2][0] * xzz_val + induced_1[0][1] *
                        induced_2[3][0] * xyz_val + induced_1[1][1] * induced_2[3][0] * yyz_val + induced_1[2][1] *
                        induced_2[3][0] * yzz_val + induced_1[0][1] * induced_2[4][0] * (3 * xzz_val - x_val) +
                        induced_1[1][1] * induced_2[4][0] * (3 * yzz_val - y_val) + induced_1[2][1] * induced_2[4][0] * (
                                    3 * zzz_val - z_val))
            a_matrix[2][5] += -(
                        induced_1[0][1] * induced_2[0][1] * (xxx_val - xyy_val) + induced_1[1][1] * induced_2[0][1] * (
                            xxy_val - yyy_val) + induced_1[2][1] * induced_2[0][1] * (xxz_val - yyz_val) + induced_1[0][1] *
                        induced_2[1][1] * xxy_val + induced_1[1][1] * induced_2[1][1] * xyy_val + induced_1[2][1] *
                        induced_2[1][1] * xyz_val + induced_1[0][1] * induced_2[2][1] * xxz_val + induced_1[1][1] *
                        induced_2[2][1] * xyz_val + induced_1[2][1] * induced_2[2][1] * xzz_val + induced_1[0][1] *
                        induced_2[3][1] * xyz_val + induced_1[1][1] * induced_2[3][1] * yyz_val + induced_1[2][1] *
                        induced_2[3][1] * yzz_val + induced_1[0][1] * induced_2[4][1] * (3 * xzz_val - x_val) +
                        induced_1[1][1] * induced_2[4][1] * (3 * yzz_val - y_val) + induced_1[2][1] * induced_2[4][1] * (
                                    3 * zzz_val - z_val))
            a_matrix[2][6] += -(
                        induced_1[0][1] * induced_2[0][2] * (xxx_val - xyy_val) + induced_1[1][1] * induced_2[0][2] * (
                            xxy_val - yyy_val) + induced_1[2][1] * induced_2[0][2] * (xxz_val - yyz_val) + induced_1[0][1] *
                        induced_2[1][2] * xxy_val + induced_1[1][1] * induced_2[1][2] * xyy_val + induced_1[2][1] *
                        induced_2[1][2] * xyz_val + induced_1[0][1] * induced_2[2][2] * xxz_val + induced_1[1][1] *
                        induced_2[2][2] * xyz_val + induced_1[2][1] * induced_2[2][2] * xzz_val + induced_1[0][1] *
                        induced_2[3][2] * xyz_val + induced_1[1][1] * induced_2[3][2] * yyz_val + induced_1[2][1] *
                        induced_2[3][2] * yzz_val + induced_1[0][1] * induced_2[4][2] * (3 * xzz_val - x_val) +
                        induced_1[1][1] * induced_2[4][2] * (3 * yzz_val - y_val) + induced_1[2][1] * induced_2[4][2] * (
                                    3 * zzz_val - z_val))
            a_matrix[2][7] += -(
                        induced_1[0][1] * induced_2[0][3] * (xxx_val - xyy_val) + induced_1[1][1] * induced_2[0][3] * (
                            xxy_val - yyy_val) + induced_1[2][1] * induced_2[0][3] * (xxz_val - yyz_val) + induced_1[0][1] *
                        induced_2[1][3] * xxy_val + induced_1[1][1] * induced_2[1][3] * xyy_val + induced_1[2][1] *
                        induced_2[1][3] * xyz_val + induced_1[0][1] * induced_2[2][3] * xxz_val + induced_1[1][1] *
                        induced_2[2][3] * xyz_val + induced_1[2][1] * induced_2[2][3] * xzz_val + induced_1[0][1] *
                        induced_2[3][3] * xyz_val + induced_1[1][1] * induced_2[3][3] * yyz_val + induced_1[2][1] *
                        induced_2[3][3] * yzz_val + induced_1[0][1] * induced_2[4][3] * (3 * xzz_val - x_val) +
                        induced_1[1][1] * induced_2[4][3] * (3 * yzz_val - y_val) + induced_1[2][1] * induced_2[4][3] * (
                                    3 * zzz_val - z_val))
            a_matrix[2][8] += -(
                        induced_1[0][1] * induced_2[0][4] * (xxx_val - xyy_val) + induced_1[1][1] * induced_2[0][4] * (
                            xxy_val - yyy_val) + induced_1[2][1] * induced_2[0][4] * (xxz_val - yyz_val) + induced_1[0][1] *
                        induced_2[1][4] * xxy_val + induced_1[1][1] * induced_2[1][4] * xyy_val + induced_1[2][1] *
                        induced_2[1][4] * xyz_val + induced_1[0][1] * induced_2[2][4] * xxz_val + induced_1[1][1] *
                        induced_2[2][4] * xyz_val + induced_1[2][1] * induced_2[2][4] * xzz_val + induced_1[0][1] *
                        induced_2[3][4] * xyz_val + induced_1[1][1] * induced_2[3][4] * yyz_val + induced_1[2][1] *
                        induced_2[3][4] * yzz_val + induced_1[0][1] * induced_2[4][4] * (3 * xzz_val - x_val) +
                        induced_1[1][1] * induced_2[4][4] * (3 * yzz_val - y_val) + induced_1[2][1] * induced_2[4][4] * (
                                    3 * zzz_val - z_val))

            a_matrix[3][3] += -((induced_1[0][2] ** 2 * xx_val) + (2 * induced_1[0][2] * induced_1[1][2] * xy_val) + (
                        2 * induced_1[0][2] * induced_1[2][2] * xz_val) + (
                                            2 * induced_1[1][2] * induced_1[2][2] * yz_val) + (
                                            induced_1[1][2] * induced_1[1][2] * yy_val) + (
                                            induced_1[2][2] * induced_1[2][2] * zz_val))
            a_matrix[3][4] += -(
                        induced_1[0][2] * induced_2[0][0] * (xxx_val - xyy_val) + induced_1[1][2] * induced_2[0][0] * (
                            xxy_val - yyy_val) + induced_1[2][2] * induced_2[0][0] * (xxz_val - yyz_val) + induced_1[0][2] *
                        induced_2[1][0] * xxy_val + induced_1[1][2] * induced_2[1][0] * xyy_val + induced_1[2][2] *
                        induced_2[1][0] * xyz_val + induced_1[0][2] * induced_2[2][0] * xxz_val + induced_1[1][2] *
                        induced_2[2][0] * xyz_val + induced_1[2][2] * induced_2[2][0] * xzz_val + induced_1[0][2] *
                        induced_2[3][0] * xyz_val + induced_1[1][2] * induced_2[3][0] * yyz_val + induced_1[2][2] *
                        induced_2[3][0] * yzz_val + induced_1[0][2] * induced_2[4][0] * (3 * xzz_val - x_val) +
                        induced_1[1][2] * induced_2[4][0] * (3 * yzz_val - y_val) + induced_1[2][2] * induced_2[4][0] * (
                                    3 * zzz_val - z_val))
            a_matrix[3][5] += -(
                        induced_1[0][2] * induced_2[0][1] * (xxx_val - xyy_val) + induced_1[1][2] * induced_2[0][1] * (
                            xxy_val - yyy_val) + induced_1[2][2] * induced_2[0][1] * (xxz_val - yyz_val) + induced_1[0][2] *
                        induced_2[1][1] * xxy_val + induced_1[1][2] * induced_2[1][1] * xyy_val + induced_1[2][2] *
                        induced_2[1][1] * xyz_val + induced_1[0][2] * induced_2[2][1] * xxz_val + induced_1[1][2] *
                        induced_2[2][1] * xyz_val + induced_1[2][2] * induced_2[2][1] * xzz_val + induced_1[0][2] *
                        induced_2[3][1] * xyz_val + induced_1[1][2] * induced_2[3][1] * yyz_val + induced_1[2][2] *
                        induced_2[3][1] * yzz_val + induced_1[0][2] * induced_2[4][1] * (3 * xzz_val - x_val) +
                        induced_1[1][2] * induced_2[4][1] * (3 * yzz_val - y_val) + induced_1[2][2] * induced_2[4][1] * (
                                    3 * zzz_val - z_val))
            a_matrix[3][6] += -(
                        induced_1[0][2] * induced_2[0][2] * (xxx_val - xyy_val) + induced_1[1][2] * induced_2[0][2] * (
                            xxy_val - yyy_val) + induced_1[2][2] * induced_2[0][2] * (xxz_val - yyz_val) + induced_1[0][2] *
                        induced_2[1][2] * xxy_val + induced_1[1][2] * induced_2[1][2] * xyy_val + induced_1[2][2] *
                        induced_2[1][2] * xyz_val + induced_1[0][2] * induced_2[2][2] * xxz_val + induced_1[1][2] *
                        induced_2[2][2] * xyz_val + induced_1[2][2] * induced_2[2][2] * xzz_val + induced_1[0][2] *
                        induced_2[3][2] * xyz_val + induced_1[1][2] * induced_2[3][2] * yyz_val + induced_1[2][2] *
                        induced_2[3][2] * yzz_val + induced_1[0][2] * induced_2[4][2] * (3 * xzz_val - x_val) +
                        induced_1[1][2] * induced_2[4][2] * (3 * yzz_val - y_val) + induced_1[2][2] * induced_2[4][2] * (
                                    3 * zzz_val - z_val))
            a_matrix[3][7] += -(
                        induced_1[0][2] * induced_2[0][3] * (xxx_val - xyy_val) + induced_1[1][2] * induced_2[0][3] * (
                            xxy_val - yyy_val) + induced_1[2][2] * induced_2[0][3] * (xxz_val - yyz_val) + induced_1[0][2] *
                        induced_2[1][3] * xxy_val + induced_1[1][2] * induced_2[1][3] * xyy_val + induced_1[2][2] *
                        induced_2[1][3] * xyz_val + induced_1[0][2] * induced_2[2][3] * xxz_val + induced_1[1][2] *
                        induced_2[2][3] * xyz_val + induced_1[2][2] * induced_2[2][3] * xzz_val + induced_1[0][2] *
                        induced_2[3][3] * xyz_val + induced_1[1][2] * induced_2[3][3] * yyz_val + induced_1[2][2] *
                        induced_2[3][3] * yzz_val + induced_1[0][2] * induced_2[4][3] * (3 * xzz_val - x_val) +
                        induced_1[1][2] * induced_2[4][3] * (3 * yzz_val - y_val) + induced_1[2][2] * induced_2[4][3] * (
                                    3 * zzz_val - z_val))
            a_matrix[3][8] += -(
                        induced_1[0][2] * induced_2[0][4] * (xxx_val - xyy_val) + induced_1[1][2] * induced_2[0][4] * (
                            xxy_val - yyy_val) + induced_1[2][2] * induced_2[0][4] * (xxz_val - yyz_val) + induced_1[0][2] *
                        induced_2[1][4] * xxy_val + induced_1[1][2] * induced_2[1][4] * xyy_val + induced_1[2][2] *
                        induced_2[1][4] * xyz_val + induced_1[0][2] * induced_2[2][4] * xxz_val + induced_1[1][2] *
                        induced_2[2][4] * xyz_val + induced_1[2][2] * induced_2[2][4] * xzz_val + induced_1[0][2] *
                        induced_2[3][4] * xyz_val + induced_1[1][2] * induced_2[3][4] * yyz_val + induced_1[2][2] *
                        induced_2[3][4] * yzz_val + induced_1[0][2] * induced_2[4][4] * (3 * xzz_val - x_val) +
                        induced_1[1][2] * induced_2[4][4] * (3 * yzz_val - y_val) + induced_1[2][2] * induced_2[4][4] * (
                                    3 * zzz_val - z_val))

            a_matrix[4][4] -= (induced_2[0][0] * induced_2[0][0] * q_11) + (
                        (induced_2[0][0] * induced_2[1][0] + induced_2[1][0] * induced_2[0][0]) * q_12) + ((induced_2[0][
                                                                                                                0] *
                                                                                                            induced_2[2][
                                                                                                                0] +
                                                                                                            induced_2[2][
                                                                                                                0] *
                                                                                                            induced_2[0][
                                                                                                                0]) * q_13) + (
                                          (induced_2[0][0] * induced_2[3][0] + induced_2[3][0] * induced_2[0][
                                              0]) * q_14) + (
                                          (induced_2[0][0] * induced_2[4][0] + induced_2[4][0] * induced_2[0][0]) * q_15)
            a_matrix[4][4] -= (induced_2[1][0] * induced_2[1][0] * q_22) + (
                        (induced_2[1][0] * induced_2[2][0] + induced_2[2][0] * induced_2[1][0]) * q_23) + ((induced_2[1][
                                                                                                                0] *
                                                                                                            induced_2[3][
                                                                                                                0] +
                                                                                                            induced_2[3][
                                                                                                                0] *
                                                                                                            induced_2[1][
                                                                                                                0]) * q_24) + (
                                          (induced_2[1][0] * induced_2[4][0] + induced_2[4][0] * induced_2[1][0]) * q_25)
            a_matrix[4][4] -= ((induced_2[2][0] * induced_2[2][0]) * q_33) + (
                        (induced_2[2][0] * induced_2[3][0] + induced_2[3][0] * induced_2[2][0]) * q_34) + (
                                          (induced_2[2][0] * induced_2[4][0] + induced_2[4][0] * induced_2[2][0]) * q_35)
            a_matrix[4][4] -= ((induced_2[3][0] * induced_2[3][0]) * q_44) + (
                        (induced_2[3][0] * induced_2[4][0] + induced_2[4][0] * induced_2[3][0]) * q_45)
            a_matrix[4][4] -= (induced_2[4][0] * induced_2[4][0] * q_55)

            a_matrix[4][5] -= (induced_2[0][0] * induced_2[0][1] * q_11) + (
                        (induced_2[0][0] * induced_2[1][1] + induced_2[1][0] * induced_2[0][1]) * q_12) + ((induced_2[0][
                                                                                                                0] *
                                                                                                            induced_2[2][
                                                                                                                1] +
                                                                                                            induced_2[2][
                                                                                                                0] *
                                                                                                            induced_2[0][
                                                                                                                1]) * q_13) + (
                                          (induced_2[0][0] * induced_2[3][1] + induced_2[3][0] * induced_2[0][
                                              1]) * q_14) + (
                                          (induced_2[0][0] * induced_2[4][1] + induced_2[4][0] * induced_2[0][1]) * q_15)
            a_matrix[4][5] -= (induced_2[1][0] * induced_2[1][1] * q_22) + (
                        (induced_2[1][0] * induced_2[2][1] + induced_2[2][0] * induced_2[1][1]) * q_23) + ((induced_2[1][
                                                                                                                0] *
                                                                                                            induced_2[3][
                                                                                                                1] +
                                                                                                            induced_2[3][
                                                                                                                0] *
                                                                                                            induced_2[1][
                                                                                                                1]) * q_24) + (
                                          (induced_2[1][0] * induced_2[4][1] + induced_2[4][0] * induced_2[1][1]) * q_25)
            a_matrix[4][5] -= ((induced_2[2][0] * induced_2[2][1]) * q_33) + (
                        (induced_2[2][0] * induced_2[3][1] + induced_2[3][0] * induced_2[2][1]) * q_34) + (
                                          (induced_2[2][0] * induced_2[4][1] + induced_2[4][0] * induced_2[2][1]) * q_35)
            a_matrix[4][5] -= ((induced_2[3][0] * induced_2[3][1]) * q_44) + (
                        (induced_2[3][0] * induced_2[4][1] + induced_2[4][0] * induced_2[3][1]) * q_45)
            a_matrix[4][5] -= (induced_2[4][0] * induced_2[4][1] * q_55)

            a_matrix[4][6] -= (induced_2[0][0] * induced_2[0][2] * q_11) + (
                        (induced_2[0][0] * induced_2[1][2] + induced_2[1][0] * induced_2[0][2]) * q_12) + ((induced_2[0][
                                                                                                                0] *
                                                                                                            induced_2[2][
                                                                                                                2] +
                                                                                                            induced_2[2][
                                                                                                                0] *
                                                                                                            induced_2[0][
                                                                                                                2]) * q_13) + (
                                          (induced_2[0][0] * induced_2[3][2] + induced_2[3][0] * induced_2[0][
                                              2]) * q_14) + (
                                          (induced_2[0][0] * induced_2[4][2] + induced_2[4][0] * induced_2[0][2]) * q_15)
            a_matrix[4][6] -= (induced_2[1][0] * induced_2[1][2] * q_22) + (
                        (induced_2[1][0] * induced_2[2][2] + induced_2[2][0] * induced_2[1][2]) * q_23) + ((induced_2[1][
                                                                                                                0] *
                                                                                                            induced_2[3][
                                                                                                                2] +
                                                                                                            induced_2[3][
                                                                                                                0] *
                                                                                                            induced_2[1][
                                                                                                                2]) * q_24) + (
                                          (induced_2[1][0] * induced_2[4][2] + induced_2[4][0] * induced_2[1][2]) * q_25)
            a_matrix[4][6] -= ((induced_2[2][0] * induced_2[2][2]) * q_33) + (
                        (induced_2[2][0] * induced_2[3][2] + induced_2[3][0] * induced_2[2][2]) * q_34) + (
                                          (induced_2[2][0] * induced_2[4][2] + induced_2[4][0] * induced_2[2][2]) * q_35)
            a_matrix[4][6] -= ((induced_2[3][0] * induced_2[3][2]) * q_44) + (
                        (induced_2[3][0] * induced_2[4][2] + induced_2[4][0] * induced_2[3][2]) * q_45)
            a_matrix[4][6] -= (induced_2[4][0] * induced_2[4][2] * q_55)

            a_matrix[4][7] -= (induced_2[0][0] * induced_2[0][3] * q_11) + (
                        (induced_2[0][0] * induced_2[1][3] + induced_2[1][0] * induced_2[0][3]) * q_12) + ((induced_2[0][
                                                                                                                0] *
                                                                                                            induced_2[2][
                                                                                                                3] +
                                                                                                            induced_2[2][
                                                                                                                0] *
                                                                                                            induced_2[0][
                                                                                                                3]) * q_13) + (
                                          (induced_2[0][0] * induced_2[3][3] + induced_2[3][0] * induced_2[0][
                                              3]) * q_14) + (
                                          (induced_2[0][0] * induced_2[4][3] + induced_2[4][0] * induced_2[0][3]) * q_15)
            a_matrix[4][7] -= (induced_2[1][0] * induced_2[1][3] * q_22) + (
                        (induced_2[1][0] * induced_2[2][3] + induced_2[2][0] * induced_2[1][3]) * q_23) + ((induced_2[1][
                                                                                                                0] *
                                                                                                            induced_2[3][
                                                                                                                3] +
                                                                                                            induced_2[3][
                                                                                                                0] *
                                                                                                            induced_2[1][
                                                                                                                3]) * q_24) + (
                                          (induced_2[1][0] * induced_2[4][3] + induced_2[4][0] * induced_2[1][3]) * q_25)
            a_matrix[4][7] -= ((induced_2[2][0] * induced_2[2][3]) * q_33) + (
                        (induced_2[2][0] * induced_2[3][3] + induced_2[3][0] * induced_2[2][3]) * q_34) + (
                                          (induced_2[2][0] * induced_2[4][3] + induced_2[4][0] * induced_2[2][3]) * q_35)
            a_matrix[4][7] -= ((induced_2[3][0] * induced_2[3][3]) * q_44) + (
                        (induced_2[3][0] * induced_2[4][3] + induced_2[4][0] * induced_2[3][3]) * q_45)
            a_matrix[4][7] -= (induced_2[4][0] * induced_2[4][3] * q_55)

            a_matrix[4][8] -= (induced_2[0][0] * induced_2[0][4] * q_11) + (
                        (induced_2[0][0] * induced_2[1][4] + induced_2[1][0] * induced_2[0][4]) * q_12) + ((induced_2[0][
                                                                                                                0] *
                                                                                                            induced_2[2][
                                                                                                                4] +
                                                                                                            induced_2[2][
                                                                                                                0] *
                                                                                                            induced_2[0][
                                                                                                                4]) * q_13) + (
                                          (induced_2[0][0] * induced_2[3][4] + induced_2[3][0] * induced_2[0][
                                              4]) * q_14) + (
                                          (induced_2[0][0] * induced_2[4][4] + induced_2[4][0] * induced_2[0][4]) * q_15)
            a_matrix[4][8] -= (induced_2[1][0] * induced_2[1][4] * q_22) + (
                        (induced_2[1][0] * induced_2[2][4] + induced_2[2][0] * induced_2[1][4]) * q_23) + ((induced_2[1][
                                                                                                                0] *
                                                                                                            induced_2[3][
                                                                                                                4] +
                                                                                                            induced_2[3][
                                                                                                                0] *
                                                                                                            induced_2[1][
                                                                                                                4]) * q_24) + (
                                          (induced_2[1][0] * induced_2[4][4] + induced_2[4][0] * induced_2[1][4]) * q_25)
            a_matrix[4][8] -= ((induced_2[2][0] * induced_2[2][4]) * q_33) + (
                        (induced_2[2][0] * induced_2[3][4] + induced_2[3][0] * induced_2[2][4]) * q_34) + (
                                          (induced_2[2][0] * induced_2[4][4] + induced_2[4][0] * induced_2[2][4]) * q_35)
            a_matrix[4][8] -= ((induced_2[3][0] * induced_2[3][4]) * q_44) + (
                        (induced_2[3][0] * induced_2[4][4] + induced_2[4][0] * induced_2[3][4]) * q_45)
            a_matrix[4][8] -= (induced_2[4][0] * induced_2[4][4] * q_55)

            a_matrix[5][5] -= (induced_2[0][1] * induced_2[0][1] * q_11) + (
                        (induced_2[0][1] * induced_2[1][1] + induced_2[1][1] * induced_2[0][1]) * q_12) + ((induced_2[0][
                                                                                                                1] *
                                                                                                            induced_2[2][
                                                                                                                1] +
                                                                                                            induced_2[2][
                                                                                                                1] *
                                                                                                            induced_2[0][
                                                                                                                1]) * q_13) + (
                                          (induced_2[0][1] * induced_2[3][1] + induced_2[3][1] * induced_2[0][
                                              1]) * q_14) + (
                                          (induced_2[0][1] * induced_2[4][1] + induced_2[4][1] * induced_2[0][1]) * q_15)
            a_matrix[5][5] -= (induced_2[1][1] * induced_2[1][1] * q_22) + (
                        (induced_2[1][1] * induced_2[2][1] + induced_2[2][1] * induced_2[1][1]) * q_23) + ((induced_2[1][
                                                                                                                1] *
                                                                                                            induced_2[3][
                                                                                                                1] +
                                                                                                            induced_2[3][
                                                                                                                1] *
                                                                                                            induced_2[1][
                                                                                                                1]) * q_24) + (
                                          (induced_2[1][1] * induced_2[4][1] + induced_2[4][1] * induced_2[1][1]) * q_25)
            a_matrix[5][5] -= ((induced_2[2][1] * induced_2[2][1]) * q_33) + (
                        (induced_2[2][1] * induced_2[3][1] + induced_2[3][1] * induced_2[2][1]) * q_34) + (
                                          (induced_2[2][1] * induced_2[4][1] + induced_2[4][1] * induced_2[2][1]) * q_35)
            a_matrix[5][5] -= ((induced_2[3][1] * induced_2[3][1]) * q_44) + (
                        (induced_2[3][1] * induced_2[4][1] + induced_2[4][1] * induced_2[3][1]) * q_45)
            a_matrix[5][5] -= (induced_2[4][1] * induced_2[4][1] * q_55)

            a_matrix[5][6] -= (induced_2[0][1] * induced_2[0][2] * q_11) + (
                        (induced_2[0][1] * induced_2[1][2] + induced_2[1][1] * induced_2[0][2]) * q_12) + ((induced_2[0][
                                                                                                                1] *
                                                                                                            induced_2[2][
                                                                                                                2] +
                                                                                                            induced_2[2][
                                                                                                                1] *
                                                                                                            induced_2[0][
                                                                                                                2]) * q_13) + (
                                          (induced_2[0][1] * induced_2[3][2] + induced_2[3][1] * induced_2[0][
                                              2]) * q_14) + (
                                          (induced_2[0][1] * induced_2[4][2] + induced_2[4][1] * induced_2[0][2]) * q_15)
            a_matrix[5][6] -= (induced_2[1][1] * induced_2[1][2] * q_22) + (
                        (induced_2[1][1] * induced_2[2][2] + induced_2[2][1] * induced_2[1][2]) * q_23) + ((induced_2[1][
                                                                                                                1] *
                                                                                                            induced_2[3][
                                                                                                                2] +
                                                                                                            induced_2[3][
                                                                                                                1] *
                                                                                                            induced_2[1][
                                                                                                                2]) * q_24) + (
                                          (induced_2[1][1] * induced_2[4][2] + induced_2[4][1] * induced_2[1][2]) * q_25)
            a_matrix[5][6] -= ((induced_2[2][1] * induced_2[2][2]) * q_33) + (
                        (induced_2[2][1] * induced_2[3][2] + induced_2[3][1] * induced_2[2][2]) * q_34) + (
                                          (induced_2[2][1] * induced_2[4][2] + induced_2[4][1] * induced_2[2][2]) * q_35)
            a_matrix[5][6] -= ((induced_2[3][1] * induced_2[3][2]) * q_44) + (
                        (induced_2[3][1] * induced_2[4][2] + induced_2[4][1] * induced_2[3][2]) * q_45)
            a_matrix[5][6] -= (induced_2[4][1] * induced_2[4][2] * q_55)

            a_matrix[5][7] -= (induced_2[0][1] * induced_2[0][3] * q_11) + (
                        (induced_2[0][1] * induced_2[1][3] + induced_2[1][1] * induced_2[0][3]) * q_12) + ((induced_2[0][
                                                                                                                1] *
                                                                                                            induced_2[2][
                                                                                                                3] +
                                                                                                            induced_2[2][
                                                                                                                1] *
                                                                                                            induced_2[0][
                                                                                                                3]) * q_13) + (
                                          (induced_2[0][1] * induced_2[3][3] + induced_2[3][1] * induced_2[0][
                                              3]) * q_14) + (
                                          (induced_2[0][1] * induced_2[4][3] + induced_2[4][1] * induced_2[0][3]) * q_15)
            a_matrix[5][7] -= (induced_2[1][1] * induced_2[1][3] * q_22) + (
                        (induced_2[1][1] * induced_2[2][3] + induced_2[2][1] * induced_2[1][3]) * q_23) + ((induced_2[1][
                                                                                                                1] *
                                                                                                            induced_2[3][
                                                                                                                3] +
                                                                                                            induced_2[3][
                                                                                                                1] *
                                                                                                            induced_2[1][
                                                                                                                3]) * q_24) + (
                                          (induced_2[1][1] * induced_2[4][3] + induced_2[4][1] * induced_2[1][3]) * q_25)
            a_matrix[5][7] -= ((induced_2[2][1] * induced_2[2][3]) * q_33) + (
                        (induced_2[2][1] * induced_2[3][3] + induced_2[3][1] * induced_2[2][3]) * q_34) + (
                                          (induced_2[2][1] * induced_2[4][3] + induced_2[4][1] * induced_2[2][3]) * q_35)
            a_matrix[5][7] -= ((induced_2[3][1] * induced_2[3][3]) * q_44) + (
                        (induced_2[3][1] * induced_2[4][3] + induced_2[4][1] * induced_2[3][3]) * q_45)
            a_matrix[5][7] -= (induced_2[4][1] * induced_2[4][3] * q_55)

            a_matrix[5][8] -= (induced_2[0][1] * induced_2[0][4] * q_11) + (
                        (induced_2[0][1] * induced_2[1][4] + induced_2[1][1] * induced_2[0][4]) * q_12) + ((induced_2[0][
                                                                                                                1] *
                                                                                                            induced_2[2][
                                                                                                                4] +
                                                                                                            induced_2[2][
                                                                                                                1] *
                                                                                                            induced_2[0][
                                                                                                                4]) * q_13) + (
                                          (induced_2[0][1] * induced_2[3][4] + induced_2[3][1] * induced_2[0][
                                              4]) * q_14) + (
                                          (induced_2[0][1] * induced_2[4][4] + induced_2[4][1] * induced_2[0][4]) * q_15)
            a_matrix[5][8] -= (induced_2[1][1] * induced_2[1][4] * q_22) + (
                        (induced_2[1][1] * induced_2[2][4] + induced_2[2][1] * induced_2[1][4]) * q_23) + ((induced_2[1][
                                                                                                                1] *
                                                                                                            induced_2[3][
                                                                                                                4] +
                                                                                                            induced_2[3][
                                                                                                                1] *
                                                                                                            induced_2[1][
                                                                                                                4]) * q_24) + (
                                          (induced_2[1][1] * induced_2[4][4] + induced_2[4][1] * induced_2[1][4]) * q_25)
            a_matrix[5][8] -= ((induced_2[2][1] * induced_2[2][4]) * q_33) + (
                        (induced_2[2][1] * induced_2[3][4] + induced_2[3][1] * induced_2[2][4]) * q_34) + (
                                          (induced_2[2][1] * induced_2[4][4] + induced_2[4][1] * induced_2[2][4]) * q_35)
            a_matrix[5][8] -= ((induced_2[3][1] * induced_2[3][4]) * q_44) + (
                        (induced_2[3][1] * induced_2[4][4] + induced_2[4][1] * induced_2[3][4]) * q_45)
            a_matrix[5][8] -= (induced_2[4][1] * induced_2[4][4] * q_55)

            a_matrix[6][6] -= (induced_2[0][2] * induced_2[0][2] * q_11) + (
                        (induced_2[0][2] * induced_2[1][2] + induced_2[1][2] * induced_2[0][2]) * q_12) + ((induced_2[0][
                                                                                                                2] *
                                                                                                            induced_2[2][
                                                                                                                2] +
                                                                                                            induced_2[2][
                                                                                                                2] *
                                                                                                            induced_2[0][
                                                                                                                2]) * q_13) + (
                                          (induced_2[0][2] * induced_2[3][2] + induced_2[3][2] * induced_2[0][
                                              2]) * q_14) + (
                                          (induced_2[0][2] * induced_2[4][2] + induced_2[4][2] * induced_2[0][2]) * q_15)
            a_matrix[6][6] -= (induced_2[1][2] * induced_2[1][2] * q_22) + (
                        (induced_2[1][2] * induced_2[2][2] + induced_2[2][2] * induced_2[1][2]) * q_23) + ((induced_2[1][
                                                                                                                2] *
                                                                                                            induced_2[3][
                                                                                                                2] +
                                                                                                            induced_2[3][
                                                                                                                2] *
                                                                                                            induced_2[1][
                                                                                                                2]) * q_24) + (
                                          (induced_2[1][2] * induced_2[4][2] + induced_2[4][2] * induced_2[1][2]) * q_25)
            a_matrix[6][6] -= ((induced_2[2][2] * induced_2[2][2]) * q_33) + (
                        (induced_2[2][2] * induced_2[3][2] + induced_2[3][2] * induced_2[2][2]) * q_34) + (
                                          (induced_2[2][2] * induced_2[4][2] + induced_2[4][2] * induced_2[2][2]) * q_35)
            a_matrix[6][6] -= ((induced_2[3][2] * induced_2[3][2]) * q_44) + (
                        (induced_2[3][2] * induced_2[4][2] + induced_2[4][2] * induced_2[3][2]) * q_45)
            a_matrix[6][6] -= (induced_2[4][2] * induced_2[4][2] * q_55)

            a_matrix[6][7] -= (induced_2[0][2] * induced_2[0][3] * q_11) + (
                        (induced_2[0][2] * induced_2[1][3] + induced_2[1][2] * induced_2[0][3]) * q_12) + ((induced_2[0][
                                                                                                                2] *
                                                                                                            induced_2[2][
                                                                                                                3] +
                                                                                                            induced_2[2][
                                                                                                                2] *
                                                                                                            induced_2[0][
                                                                                                                3]) * q_13) + (
                                          (induced_2[0][2] * induced_2[3][3] + induced_2[3][2] * induced_2[0][
                                              3]) * q_14) + (
                                          (induced_2[0][2] * induced_2[4][3] + induced_2[4][2] * induced_2[0][3]) * q_15)
            a_matrix[6][7] -= (induced_2[1][2] * induced_2[1][3] * q_22) + (
                        (induced_2[1][2] * induced_2[2][3] + induced_2[2][2] * induced_2[1][3]) * q_23) + ((induced_2[1][
                                                                                                                2] *
                                                                                                            induced_2[3][
                                                                                                                3] +
                                                                                                            induced_2[3][
                                                                                                                2] *
                                                                                                            induced_2[1][
                                                                                                                3]) * q_24) + (
                                          (induced_2[1][2] * induced_2[4][3] + induced_2[4][2] * induced_2[1][3]) * q_25)
            a_matrix[6][7] -= ((induced_2[2][2] * induced_2[2][3]) * q_33) + (
                        (induced_2[2][2] * induced_2[3][3] + induced_2[3][2] * induced_2[2][3]) * q_34) + (
                                          (induced_2[2][2] * induced_2[4][3] + induced_2[4][2] * induced_2[2][3]) * q_35)
            a_matrix[6][7] -= ((induced_2[3][2] * induced_2[3][3]) * q_44) + (
                        (induced_2[3][2] * induced_2[4][3] + induced_2[4][2] * induced_2[3][3]) * q_45)
            a_matrix[6][7] -= (induced_2[4][2] * induced_2[4][3] * q_55)

            a_matrix[6][8] -= (induced_2[0][2] * induced_2[0][4] * q_11) + (
                        (induced_2[0][2] * induced_2[1][4] + induced_2[1][2] * induced_2[0][4]) * q_12) + ((induced_2[0][
                                                                                                                2] *
                                                                                                            induced_2[2][
                                                                                                                4] +
                                                                                                            induced_2[2][
                                                                                                                2] *
                                                                                                            induced_2[0][
                                                                                                                4]) * q_13) + (
                                          (induced_2[0][2] * induced_2[3][4] + induced_2[3][2] * induced_2[0][
                                              4]) * q_14) + (
                                          (induced_2[0][2] * induced_2[4][4] + induced_2[4][2] * induced_2[0][4]) * q_15)
            a_matrix[6][8] -= (induced_2[1][2] * induced_2[1][4] * q_22) + (
                        (induced_2[1][2] * induced_2[2][4] + induced_2[2][2] * induced_2[1][4]) * q_23) + ((induced_2[1][
                                                                                                                2] *
                                                                                                            induced_2[3][
                                                                                                                4] +
                                                                                                            induced_2[3][
                                                                                                                2] *
                                                                                                            induced_2[1][
                                                                                                                4]) * q_24) + (
                                          (induced_2[1][2] * induced_2[4][4] + induced_2[4][2] * induced_2[1][4]) * q_25)
            a_matrix[6][8] -= ((induced_2[2][2] * induced_2[2][4]) * q_33) + (
                        (induced_2[2][2] * induced_2[3][4] + induced_2[3][2] * induced_2[2][4]) * q_34) + (
                                          (induced_2[2][2] * induced_2[4][4] + induced_2[4][2] * induced_2[2][4]) * q_35)
            a_matrix[6][8] -= ((induced_2[3][2] * induced_2[3][4]) * q_44) + (
                        (induced_2[3][2] * induced_2[4][4] + induced_2[4][2] * induced_2[3][4]) * q_45)
            a_matrix[6][8] -= (induced_2[4][2] * induced_2[4][4] * q_55)

            a_matrix[7][7] -= (induced_2[0][3] * induced_2[0][3] * q_11) + (
                        (induced_2[0][3] * induced_2[1][3] + induced_2[1][3] * induced_2[0][3]) * q_12) + ((induced_2[0][
                                                                                                                3] *
                                                                                                            induced_2[2][
                                                                                                                3] +
                                                                                                            induced_2[2][
                                                                                                                3] *
                                                                                                            induced_2[0][
                                                                                                                3]) * q_13) + (
                                          (induced_2[0][3] * induced_2[3][3] + induced_2[3][3] * induced_2[0][
                                              3]) * q_14) + (
                                          (induced_2[0][3] * induced_2[4][3] + induced_2[4][3] * induced_2[0][3]) * q_15)
            a_matrix[7][7] -= (induced_2[1][3] * induced_2[1][3] * q_22) + (
                        (induced_2[1][3] * induced_2[2][3] + induced_2[2][3] * induced_2[1][3]) * q_23) + ((induced_2[1][
                                                                                                                3] *
                                                                                                            induced_2[3][
                                                                                                                3] +
                                                                                                            induced_2[3][
                                                                                                                3] *
                                                                                                            induced_2[1][
                                                                                                                3]) * q_24) + (
                                          (induced_2[1][3] * induced_2[4][3] + induced_2[4][3] * induced_2[1][3]) * q_25)
            a_matrix[7][7] -= ((induced_2[2][3] * induced_2[2][3]) * q_33) + (
                        (induced_2[2][3] * induced_2[3][3] + induced_2[3][3] * induced_2[2][3]) * q_34) + (
                                          (induced_2[2][3] * induced_2[4][3] + induced_2[4][3] * induced_2[2][3]) * q_35)
            a_matrix[7][7] -= ((induced_2[3][3] * induced_2[3][3]) * q_44) + (
                        (induced_2[3][3] * induced_2[4][3] + induced_2[4][3] * induced_2[3][3]) * q_45)
            a_matrix[7][7] -= (induced_2[4][3] * induced_2[4][3] * q_55)

            a_matrix[7][8] -= (induced_2[0][3] * induced_2[0][4] * q_11) + (
                        (induced_2[0][3] * induced_2[1][4] + induced_2[1][3] * induced_2[0][4]) * q_12) + ((induced_2[0][
                                                                                                                3] *
                                                                                                            induced_2[2][
                                                                                                                4] +
                                                                                                            induced_2[2][
                                                                                                                3] *
                                                                                                            induced_2[0][
                                                                                                                4]) * q_13) + (
                                          (induced_2[0][3] * induced_2[3][4] + induced_2[3][3] * induced_2[0][
                                              4]) * q_14) + (
                                          (induced_2[0][3] * induced_2[4][4] + induced_2[4][3] * induced_2[0][4]) * q_15)
            a_matrix[7][8] -= (induced_2[1][3] * induced_2[1][4] * q_22) + (
                        (induced_2[1][3] * induced_2[2][4] + induced_2[2][3] * induced_2[1][4]) * q_23) + ((induced_2[1][
                                                                                                                3] *
                                                                                                            induced_2[3][
                                                                                                                4] +
                                                                                                            induced_2[3][
                                                                                                                3] *
                                                                                                            induced_2[1][
                                                                                                                4]) * q_24) + (
                                          (induced_2[1][3] * induced_2[4][4] + induced_2[4][3] * induced_2[1][4]) * q_25)
            a_matrix[7][8] -= ((induced_2[2][3] * induced_2[2][4]) * q_33) + (
                        (induced_2[2][3] * induced_2[3][4] + induced_2[3][3] * induced_2[2][4]) * q_34) + (
                                          (induced_2[2][3] * induced_2[4][4] + induced_2[4][3] * induced_2[2][4]) * q_35)
            a_matrix[7][8] -= ((induced_2[3][3] * induced_2[3][4]) * q_44) + (
                        (induced_2[3][3] * induced_2[4][4] + induced_2[4][3] * induced_2[3][4]) * q_45)
            a_matrix[7][8] -= (induced_2[4][3] * induced_2[4][4] * q_55)

            a_matrix[8][8] -= (induced_2[0][4] * induced_2[0][4] * q_11) + (
                        (induced_2[0][4] * induced_2[1][4] + induced_2[1][4] * induced_2[0][4]) * q_12) + ((induced_2[0][
                                                                                                                4] *
                                                                                                            induced_2[2][
                                                                                                                4] +
                                                                                                            induced_2[2][
                                                                                                                4] *
                                                                                                            induced_2[0][
                                                                                                                4]) * q_13) + (
                                          (induced_2[0][4] * induced_2[3][4] + induced_2[3][4] * induced_2[0][
                                              4]) * q_14) + (
                                          (induced_2[0][4] * induced_2[4][4] + induced_2[4][4] * induced_2[0][4]) * q_15)
            a_matrix[8][8] -= (induced_2[1][4] * induced_2[1][4] * q_22) + (
                        (induced_2[1][4] * induced_2[2][4] + induced_2[2][4] * induced_2[1][4]) * q_23) + ((induced_2[1][
                                                                                                                4] *
                                                                                                            induced_2[3][
                                                                                                                4] +
                                                                                                            induced_2[3][
                                                                                                                4] *
                                                                                                            induced_2[1][
                                                                                                                4]) * q_24) + (
                                          (induced_2[1][4] * induced_2[4][4] + induced_2[4][4] * induced_2[1][4]) * q_25)
            a_matrix[8][8] -= ((induced_2[2][4] * induced_2[2][4]) * q_33) + (
                        (induced_2[2][4] * induced_2[3][4] + induced_2[3][4] * induced_2[2][4]) * q_34) + (
                                          (induced_2[2][4] * induced_2[4][4] + induced_2[4][4] * induced_2[2][4]) * q_35)
            a_matrix[8][8] -= ((induced_2[3][4] * induced_2[3][4]) * q_44) + (
                        (induced_2[3][4] * induced_2[4][4] + induced_2[4][4] * induced_2[3][4]) * q_45)
            a_matrix[8][8] -= (induced_2[4][4] * induced_2[4][4] * q_55)

    for i_spec in range(0, 9):
        for j_spec in range(0, i_spec):
            a_matrix[i_spec, j_spec] = a_matrix[j_spec, i_spec]

    return np.real(a_matrix)
