"""
script of helper functions
"""
import numpy as np
import math
from scipy.spatial import distance


def get_chain(no_atoms, level_mat, adjacency_matrix, sphere, level):
    """
    Inputs the level matrix, Intersection matrix and writes the path from base sphere
    @param no_atoms:
    @param level_mat:
    @param adjacency_matrix:
    @param sphere:
    @param level:
    @return: chain - path through molecule from base sphere
    """
    chain = np.zeros(
        level + 1, dtype=int
    )  # whatever the level is we will output a vector of length L+1
    i = level
    current_sphere = sphere
    chain[level] = sphere

    while i > 0:
        for k in range(0, no_atoms):
            # if there is a lower level non-t sphere and it meets the current sphere
            if level_mat[i - 1][k] == 1 and adjacency_matrix[current_sphere][k] == 1:
                current_sphere = k
                chain[i - 1] = k
        i = i - 1

    return chain


def get_m_rot(vector):
    """
    the element of SO(3) that carries out the rotation
    @param vector:
    @return: m_rot
    """
    m_rot = np.array(
        [
            [
                1 - ((vector[0] ** 2) / (1 + vector[2])),
                -vector[0] * vector[1] / (1 + vector[2]),
                vector[0],
            ],
            [
                -vector[0] * vector[1] / (1 + vector[2]),
                1 - ((vector[1] ** 2) / (1 + vector[2])),
                vector[1],
            ],
            [-vector[0], -vector[1], vector[2]],
        ]
    )

    return m_rot


# helper functions for performing piecewise stereographic projection
# if we rotate (0,0,1) onto (v_1,v_2,v_3) this induces an element of PSU(2)  [[alpha, beta],[-conj(beta), alpha]]
def alpha_coefficient(vector):  # alpha coefficient
    """
    function to get alpha coefficient
    @param vector:
    @return: alpha
    """
    if (vector[2] + 1) ** 2 > 10 ** (-9):
        return math.sqrt((1 + vector[2]) / 2)
    else:
        return 0


def beta_coefficient(vector):  # beta coefficient
    """
    function to get beta coefficient
    @param vector:
    @return: beta
    """
    if (vector[2] + 1) ** 2 > 10 ** (-9):
        return -math.sqrt(1 / (2 * (1 + vector[2]))) * complex(vector[0], vector[1])
    else:
        return 1j


def t_circle(alpha, beta, gamma, delta, c, r_rel):
    """
    function that computes the centre and radius of the image of a circle under the Mobius transformation
    z-->(az+b)/(cz+d)
    @param alpha:
    @param beta:
    @param gamma:
    @param delta:
    @param c:
    @param r_rel:
    @return: [cent, Radius]
    """
    cent = (
        ((beta + (c * alpha)) * np.conj(delta + (c * gamma)))
        - (r_rel * r_rel * alpha * np.conj(gamma))
    ) / ((abs(delta + (c * gamma)) ** 2) - (r_rel * abs(gamma)) ** 2)
    k = (((r_rel * abs(alpha)) ** 2) - (abs(beta + (c * alpha))) ** 2) / (
        (abs(delta + (c * gamma)) ** 2) - (r_rel * abs(gamma)) ** 2
    )
    radius = math.sqrt(abs(k + abs(cent) ** 2))

    return [cent, radius]


def transform_v2(a, r):
    """
    @param a:
    @param r:
    @return: transform
    """
    alpha = 1
    beta = (
        (abs(a) ** 2 - r ** 2 - 1)
        + math.sqrt((abs(a) ** 2 - r ** 2 - 1) ** 2 + 4 * abs(a) ** 2)
    ) / (-2 * np.conj(a))
    scale = 1 + abs(beta) ** 2
    alpha = alpha / np.sqrt(scale)
    beta = beta / np.sqrt(scale)

    transform = np.array([[alpha, beta], [-np.conj(beta), alpha]])

    if abs((-beta / alpha) - a) > r:
        transform = np.array([[np.conj(beta), alpha], [-alpha, beta]])

    return transform


def cut_10(inputs, error):
    """
    Function to cut out levels > 10 (removes "crunching" of spheres that triggers LinAlgError)
    @param inputs:
    @param error:
    @return: updated inputs with > level 10 atoms omitted
    """
    # unpack tuples
    no_atoms = inputs.no_atoms
    adjacency_matrix = inputs.adjacency_matrix
    radii = inputs.radii
    centres = inputs.centres
    sphere_levels_vec = error.sphere_levels_vec

    keep = []  # vector telling us whether to keep (1) or cut (0) sphere

    for sphere in range(0, no_atoms):
        level = sphere_levels_vec[sphere]
        if level < 11:
            keep.append(1)
        else:
            keep.append(0)
    size = sum(keep)

    new_centres = np.zeros((size, 3), dtype=float)
    new_radii = np.zeros(size, dtype=float)
    adj_mat_2 = np.zeros((size, size), dtype=int)

    ind_i = 0
    for sphere in range(0, no_atoms):
        if keep[sphere] == 1:
            new_centres[ind_i][0] = centres[sphere][0]
            new_centres[ind_i][1] = centres[sphere][1]
            new_centres[ind_i][2] = centres[sphere][2]
            new_radii[ind_i] = radii[sphere]
            ind_j = 0
            for sphere_2 in range(0, no_atoms):
                if keep[sphere_2] == 1:
                    adj_mat_2[ind_i, ind_j] = adjacency_matrix[sphere, sphere_2]
                    ind_j += 1
            ind_i += 1

    adjacency_matrix = adj_mat_2

    new_inputs = namedtuple(
        "input", ["no_atoms", "radii", "centres", "adjacency_matrix"]
    )

    return new_inputs(
        no_atoms=size,
        radii=new_radii,
        centres=new_centres,
        adjacency_matrix=adjacency_matrix,
    )


def get_score(query, test, query_id=None, test_id=None):
    """
    Function to compute the normalised Bray-Curtis distance between two molecules
    @param query:
    @param test:
    @param query_id:
    @param test_id:
    @return: "self" if the query and test ids match, score rounded to 3dp
    """
    if query_id is not None:
        if query_id == test_id:
            return "self"  # marker for self comparison
        else:
            return round((1 - distance.braycurtis(query, test)), 3)  # return score to 3dp
    else:
        return round((1 - distance.braycurtis(query, test)), 3)  # return score to 3dp


def vol_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return: vol_integral
    """
    a_conj = np.conj(a)
    k_0 = a * a_conj
    k_1 = np.sqrt((k_0 + b) ** 2)
    k_2 = np.sqrt((k_0 + b) ** 2 - 2 * k_0 * x ** 2 + 2 * b * x ** 2 + x ** 4)

    ret = (1 / 4) * (
        1 / k_1 - 1 / k_2 + x ** 2 / (b * k_2) + (k_0 * (1 / k_1 - 1 / k_2)) / b
    )

    return 4 * np.pi * c * ret


def x_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return: x_integral
    """
    a_conj = np.conj(a)

    # helper variables to tidy up function
    k_0 = a * a_conj
    k_1 = np.sqrt((k_0 + b) ** 2)
    k_2 = np.sqrt((k_0 + b) ** 2 - 2 * k_0 * x ** 2 + 2 * b * x ** 2 + x ** 4)
    k_3 = a ** 2 * a_conj ** 2

    if abs(a_conj) >= 0.00000001:
        ret = (
            (a + a_conj)
            * (
                np.sqrt(1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2)
                * (
                    a ** 3 * a_conj ** 3
                    + k_3 * (1 + b - x ** 2)
                    + (-1 + b) * (-(b ** 2) - b * x ** 2 + k_1 * k_2)
                    - k_0 * (b ** 2 + x ** 2 + 2 * b * (-1 + x ** 2) + k_1 * k_2)
                )
                + 4
                * k_0
                * b
                * k_2
                * np.arctanh(
                    (k_3 + (-1 + b) * b + a * (a_conj + 2 * a_conj * b))
                    / (k_1 * np.sqrt(k_3 + (-1 + b) ** 2 + 2 * k_0 * (1 + b)))
                )
                - 4
                * k_0
                * b
                * k_2
                * np.arctanh(
                    (
                        k_3
                        + b ** 2
                        - x ** 2
                        + k_0 * (1 + 2 * b - x ** 2)
                        + b * (-1 + x ** 2)
                    )
                    / (
                        np.sqrt(k_3 + (-1 + b) ** 2 + 2 * k_0 * (1 + b))
                        * np.sqrt(k_3 + 2 * k_0 * (b - x ** 2) + (b + x ** 2) ** 2)
                    )
                )
            )
        ) / (
            2
            * k_0
            * b
            * (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** (3 / 2)
            * np.sqrt(k_3 + 2 * k_0 * (b - x ** 2) + (b + x ** 2) ** 2)
        )

        return -2 * np.pi * c * ret

    else:

        return 0


def z_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return: z_integral
    """
    a_conj = np.conj(a)

    # helper variables to tidy up function
    k_0 = a * a_conj
    k_1 = np.sqrt((k_0 + b) ** 2)
    k_2 = np.sqrt((k_0 + b) ** 2 - 2 * k_0 * x ** 2 + 2 * b * x ** 2 + x ** 4)
    k_3 = a ** 2 * a_conj ** 2
    k_4 = np.sqrt(1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2)
    k_12 = k_1 * k_2
    k_14 = k_1 * k_4
    k_24 = k_2 * k_4

    if abs(a_conj) >= 0.00000001:
        ret = (
            (-a) * a_conj * k_14
            + a ** 3 * a_conj ** 3 * k_14
            - b * k_14
            + 4 * k_0 * b * k_14
            + 3 * k_3 * b * k_14
            + 4 * b ** 2 * k_14
            + 3 * k_0 * b ** 2 * k_14
            + b ** 3 * k_14
            + k_14 * x ** 2
            - k_3 * k_14 * x ** 2
            + 4 * b * k_14 * x ** 2
            - 2 * k_0 * b * k_14 * x ** 2
            - b ** 2 * k_14 * x ** 2
            + k_0 * k_24
            - a ** 3 * a_conj ** 3 * k_24
            + b * k_24
            - 4 * k_0 * b * k_24
            - 3 * k_3 * b * k_24
            - 4 * b ** 2 * k_24
            - 3 * k_0 * b ** 2 * k_24
            - b ** 3 * k_24
            + 4 * b * (-1 + k_0 + b) * k_12 * np.log(k_0 - b + (k_0 + b) ** 2 + k_14)
            + 4 * b * (-1 + k_0 + b) * k_12 * np.log(1 + x ** 2)
            + 4
            * b
            * k_12
            * np.log(
                k_3
                - b
                + b ** 2
                - x ** 2
                + b * x ** 2
                + k_0 * (1 + 2 * b - x ** 2)
                + np.sqrt(k_3 + (-1 + b) ** 2 + 2 * k_0 * (1 + b))
                * np.sqrt(k_3 + 2 * k_0 * (b - x ** 2) + (b + x ** 2) ** 2)
            )
            - 4
            * k_0
            * b
            * k_1
            * k_2
            * np.log(
                k_3
                - b
                + b ** 2
                - x ** 2
                + b * x ** 2
                + k_0 * (1 + 2 * b - x ** 2)
                + np.sqrt(k_3 + (-1 + b) ** 2 + 2 * k_0 * (1 + b))
                * np.sqrt(k_3 + 2 * k_0 * (b - x ** 2) + (b + x ** 2) ** 2)
            )
            - 4
            * b ** 2
            * k_12
            * np.log(
                k_3
                - b
                + b ** 2
                - x ** 2
                + b * x ** 2
                + k_0 * (1 + 2 * b - x ** 2)
                + np.sqrt(k_3 + (-1 + b) ** 2 + 2 * k_0 * (1 + b))
                * np.sqrt(k_3 + 2 * k_0 * (b - x ** 2) + (b + x ** 2) ** 2)
            )
        ) / (
            4
            * b
            * k_1
            * (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** (3 / 2)
            * np.sqrt(k_3 + 2 * k_0 * (b - x ** 2) + (b + x ** 2) ** 2)
        )

        return 4 * np.pi * c * ret

    if abs(a_conj) < 0.00000001 and (b - 1) ** 2 > np.sqrt(0.00000001):
        ret = (
            x ** 2
            - b ** 2 * x ** 2
            + 2 * b * (b + x ** 2) * np.log(b)
            + 2 * b * (b + x ** 2) * np.log(1 + x ** 2)
            - 2 * b ** 2 * np.log(b + x ** 2)
            - 2 * b * x ** 2 * np.log(b + x ** 2)
        ) / (2 * (-1 + b) ** 2 * b * (b + x ** 2))

        return 4 * np.pi * c * ret

    if abs(a_conj) < 0.00000001 and (b - 1) ** 2 <= np.sqrt(0.00000001):
        ret = 0.5 * x ** 2 / (1 + x ** 2) ** 2

        return 4 * np.pi * c * ret


def xx_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return: xx_integral
    """
    a_conj = np.conj(a)

    k_1 = np.sqrt((a * a_conj + b) ** 2)
    k_2 = np.sqrt(
        (a * a_conj + b) ** 2 - 2 * a * a_conj * x ** 2 + 2 * b * x ** 2 + x ** 4
    )
    k_3 = np.sqrt(a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b))

    if abs(a_conj) >= 0.00000001:
        ret = (
            np.sqrt(1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2)
            * (
                a ** 7 * a_conj ** 5 * (-1 + b - x ** 2)
                + a ** 6
                * a_conj ** 4
                * (
                    -2
                    + 3 * b
                    + 5 * b ** 2
                    - x ** 2
                    - 2 * b * x ** 2
                    + x ** 4
                    - 2 * a_conj ** 2 * (1 + x ** 2)
                    + b * x ** 2 * k_2
                )
                - a_conj ** 2
                * (-1 + b) ** 3
                * b
                * (
                    -(b ** 2)
                    - x ** 4
                    + k_1 * k_2
                    + (1 + k_1) * x ** 2 * k_2
                    - b * x ** 2 * (2 + k_2)
                )
                + a ** 5
                * a_conj ** 3
                * (
                    10 * b ** 3
                    + a_conj ** 4 * (-1 + b - x ** 2)
                    + 2 * a_conj ** 2 * (-2 + 2 * b - x ** 2 + x ** 4)
                    + (1 + x ** 2) * (-1 + 2 * x ** 2 + k_1 * k_2)
                    + 2 * b ** 2 * (6 + x ** 2 + 2 * x ** 2 * k_2)
                    - b
                    * (9 - 4 * x ** 4 + k_1 * k_2 + x ** 2 * (9 - 4 * k_2 + k_1 * k_2))
                )
                + a ** 4
                * a_conj ** 2
                * (
                    10 * b ** 4
                    + (1 + x ** 2) * (x ** 2 + 2 * k_1 * k_2)
                    + a_conj ** 4
                    * (
                        -2
                        + 3 * b
                        + 5 * b ** 2
                        - x ** 2
                        - 2 * b * x ** 2
                        + x ** 4
                        + b * x ** 2 * k_2
                    )
                    + b ** 3 * (8 + x ** 2 * (8 + 6 * k_2))
                    + 2
                    * a_conj ** 2
                    * (
                        6 * b ** 2 * (2 + x ** 2)
                        + b * (-8 - 7 * x ** 2 + 3 * x ** 4)
                        + (1 + x ** 2) * (-1 + 2 * x ** 2 + k_1 * k_2)
                    )
                    - b ** 2
                    * (
                        -6 * x ** 4
                        + 3 * (3 + k_1 * k_2)
                        + x ** 2 * (-3 - 4 * k_2 + 3 * k_1 * k_2)
                    )
                    - b
                    * (
                        3
                        - 7 * x ** 4
                        + 5 * k_1 * k_2
                        + x ** 2 * (-8 - 6 * k_2 + 5 * k_1 * k_2)
                    )
                )
                - a
                * a_conj ** 3
                * (-1 + b)
                * (
                    -5 * b ** 4
                    + k_1 * (1 + x ** 2) * k_2
                    - b ** 3 * (2 + x ** 2 * (7 + 4 * k_2))
                    + b ** 2
                    * (
                        -4 * x ** 4
                        + 3 * x ** 2 * (-4 + k_1 * k_2)
                        + 3 * (-1 + k_1 * k_2)
                    )
                    + b
                    * (
                        -6 * x ** 4
                        + 6 * k_1 * k_2
                        + x ** 2 * (-1 + 4 * k_2 + 6 * k_1 * k_2)
                    )
                )
                - a ** 2
                * (
                    (-1 + b) ** 3
                    * b
                    * (
                        -(b ** 2)
                        - x ** 4
                        + k_1 * k_2
                        + (1 + k_1) * x ** 2 * k_2
                        - b * x ** 2 * (2 + k_2)
                    )
                    + 2
                    * a_conj ** 2
                    * (-1 + b)
                    * (
                        -(b ** 3) * (5 + 3 * x ** 2)
                        - b ** 2 * (1 + 6 * x ** 2 + x ** 4)
                        + k_1 * (1 + x ** 2) * k_2
                        + b
                        * (-5 * x ** 4 + 5 * k_1 * k_2 + x ** 2 * (-3 + 5 * k_1 * k_2))
                    )
                    + a_conj ** 4
                    * (
                        -10 * b ** 4
                        - (1 + x ** 2) * (x ** 2 + 2 * k_1 * k_2)
                        - 2 * b ** 3 * (4 + x ** 2 * (4 + 3 * k_2))
                        + b ** 2
                        * (
                            -6 * x ** 4
                            + 3 * (3 + k_1 * k_2)
                            + x ** 2 * (-3 - 4 * k_2 + 3 * k_1 * k_2)
                        )
                        + b
                        * (
                            3
                            - 7 * x ** 4
                            + 5 * k_1 * k_2
                            + x ** 2 * (-8 - 6 * k_2 + 5 * k_1 * k_2)
                        )
                    )
                )
                - a ** 3
                * a_conj
                * (
                    a_conj ** 4
                    * (
                        -10 * b ** 3
                        - (1 + x ** 2) * (-1 + 2 * x ** 2 + k_1 * k_2)
                        - 2 * b ** 2 * (6 + x ** 2 + 2 * x ** 2 * k_2)
                        + b
                        * (
                            9
                            - 4 * x ** 4
                            + k_1 * k_2
                            + x ** 2 * (9 - 4 * k_2 + k_1 * k_2)
                        )
                    )
                    + 2
                    * a_conj ** 2
                    * (
                        -2 * b ** 3 * (7 + 4 * x ** 2)
                        + b ** 2 * (10 + 3 * x ** 2 - 3 * x ** 4)
                        - (1 + x ** 2) * (x ** 2 + 2 * k_1 * k_2)
                        + 2
                        * b
                        * (
                            1
                            - 3 * x ** 4
                            + 2 * k_1 * k_2
                            + x ** 2 * (-3 + 2 * k_1 * k_2)
                        )
                    )
                    + (-1 + b)
                    * (
                        -5 * b ** 4
                        + k_1 * (1 + x ** 2) * k_2
                        - b ** 3 * (2 + x ** 2 * (7 + 4 * k_2))
                        + b ** 2
                        * (
                            -4 * x ** 4
                            + 3 * x ** 2 * (-4 + k_1 * k_2)
                            + 3 * (-1 + k_1 * k_2)
                        )
                        + b
                        * (
                            -6 * x ** 4
                            + 6 * k_1 * k_2
                            + x ** 2 * (-1 + 4 * k_2 + 6 * k_1 * k_2)
                        )
                    )
                )
            )
            + 4
            * a ** 2
            * a_conj ** 2
            * b
            * (
                a ** 3 * a_conj * (-3 + a_conj ** 2)
                + 3 * a ** 2 * (-1 + a_conj ** 2 * (-1 + b) - b)
                + (-3 * a_conj ** 2 + (-1 + b) ** 2) * (1 + b)
                - a * a_conj * (3 + 3 * a_conj ** 2 + 4 * b - 3 * b ** 2)
            )
            * (1 + x ** 2)
            * k_2
            * np.arctanh(
                (a ** 2 * a_conj ** 2 + (-1 + b) * b + a * (a_conj + 2 * a_conj * b))
                / (k_1 * k_3)
            )
            - 4
            * a ** 2
            * a_conj ** 2
            * b
            * (
                a ** 3 * a_conj * (-3 + a_conj ** 2)
                + 3 * a ** 2 * (-1 + a_conj ** 2 * (-1 + b) - b)
                + (-3 * a_conj ** 2 + (-1 + b) ** 2) * (1 + b)
                - a * a_conj * (3 + 3 * a_conj ** 2 + 4 * b - 3 * b ** 2)
            )
            * (1 + x ** 2)
            * k_2
            * np.arctanh(
                (
                    a ** 2 * a_conj ** 2
                    + a * a_conj * (1 + 2 * b - x ** 2)
                    + (-1 + b) * (b + x ** 2)
                )
                / (
                    k_3
                    * np.sqrt(
                        a ** 2 * a_conj ** 2
                        + 2 * a * a_conj * (b - x ** 2)
                        + (b + x ** 2) ** 2
                    )
                )
            )
        ) / (
            2
            * a ** 2
            * a_conj ** 2
            * b
            * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (5 / 2)
            * (1 + x ** 2)
            * np.sqrt(
                a ** 2 * a_conj ** 2 + 2 * a * a_conj * (b - x ** 2) + (b + x ** 2) ** 2
            )
        )
        return 2 * np.pi * c * ret
    if abs(a_conj) <= 0.00000001 and (b - 1) ** 2 > np.sqrt(0.00000001):
        ret = (
            2
            - 2 * b
            - 1 / (1 + x ** 2)
            + b / (1 + x ** 2)
            - b / (b + x ** 2)
            + b ** 2 / (b + x ** 2)
            + (1 + b) * np.log(b)
            + (1 + b) * np.log(1 + x ** 2)
            - np.log(b + x ** 2)
            - b * np.log(b + x ** 2)
        ) / (-1 + b) ** 3
        return 4 * np.pi * c * ret
    if abs(a_conj) <= 0.00000001 and (b - 1) ** 2 <= np.sqrt(0.00000001):
        ret = x ** 4 * (3 + x ** 2) / (6 * (1 + x ** 2) ** 3)
        return 4 * np.pi * c * ret


def xy_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return:
    """
    a_conj = np.conj(a)

    # helper variables to tidy up function
    k_0 = a * a_conj
    k_1 = np.sqrt((k_0 + b) ** 2)
    k_2 = np.sqrt((k_0 + b) ** 2 - 2 * k_0 * x ** 2 + 2 * b * x ** 2 + x ** 4)
    k_3 = a ** 2 * a_conj ** 2
    k_12 = k_1 * k_2

    if abs(a_conj) >= 0.00000001:
        ret = (1 / (2 * (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** (5 / 2))) * (
            (a ** 2 - a_conj ** 2)
            * (
                (
                    np.sqrt(1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2)
                    * (
                        a ** 5 * a_conj ** 5 * (-1 + b - x ** 2)
                        + a ** 4
                        * a_conj ** 4
                        * (
                            -2
                            + 3 * b
                            + 5 * b ** 2
                            - x ** 2
                            - 2 * b * x ** 2
                            + x ** 4
                            + b * x ** 2 * k_2
                        )
                        - (-1 + b) ** 3
                        * b
                        * (
                            -(b ** 2)
                            - x ** 4
                            + k_12
                            + (1 + k_1) * x ** 2 * k_2
                            - b * x ** 2 * (2 + k_2)
                        )
                        - a ** 3
                        * a_conj ** 3
                        * (
                            -10 * b ** 3
                            - (1 + x ** 2) * (-1 + 2 * x ** 2 + k_12)
                            - 2 * b ** 2 * (6 + x ** 2 + 2 * x ** 2 * k_2)
                            + b
                            * (9 - 4 * x ** 4 + k_12 + x ** 2 * (9 - 4 * k_2 + k_12))
                        )
                        - k_3
                        * (
                            -10 * b ** 4
                            - (1 + x ** 2) * (x ** 2 + 2 * k_12)
                            - 2 * b ** 3 * (4 + x ** 2 * (4 + 3 * k_2))
                            + b ** 2
                            * (
                                -6 * x ** 4
                                + 3 * (3 + k_12)
                                + x ** 2 * (-3 - 4 * k_2 + 3 * k_12)
                            )
                            + b
                            * (
                                3
                                - 7 * x ** 4
                                + 5 * k_12
                                + x ** 2 * (-8 - 6 * k_2 + 5 * k_12)
                            )
                        )
                        - k_0
                        * (-1 + b)
                        * (
                            -5 * b ** 4
                            + k_1 * (1 + x ** 2) * k_2
                            - b ** 3 * (2 + x ** 2 * (7 + 4 * k_2))
                            + b ** 2
                            * (-4 * x ** 4 + 3 * x ** 2 * (-4 + k_12) + 3 * (-1 + k_12))
                            + b
                            * (
                                -6 * x ** 4
                                + 6 * k_12
                                + x ** 2 * (-1 + 4 * k_2 + 6 * k_12)
                            )
                        )
                    )
                )
                / (k_3 * b * (1 + x ** 2) * k_2)
                - 12
                * (1 + k_0 + b)
                * np.log(
                    k_0
                    - b
                    + (k_0 + b) ** 2
                    + k_1 * np.sqrt(1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2)
                )
                - 12 * (1 + k_0 + b) * np.log(1 + x ** 2)
                + 12
                * (1 + k_0 + b)
                * np.log(
                    k_3
                    - b
                    + b ** 2
                    - x ** 2
                    + b * x ** 2
                    + k_0 * (1 + 2 * b - x ** 2)
                    + np.sqrt(k_3 + (-1 + b) ** 2 + 2 * k_0 * (1 + b))
                    * np.sqrt(k_3 + 2 * k_0 * (b - x ** 2) + (b + x ** 2) ** 2)
                )
            )
        )
        return (2 / 1j) * np.pi * c * ret
    else:
        return 0


def xz_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return:
    """
    a_conj = np.conj(a)

    # helper variables to tidy up function
    k_0 = a * a_conj
    k_1 = np.sqrt((k_0 + b) ** 2)
    k_2 = np.sqrt((k_0 + b) ** 2 - 2 * k_0 * x ** 2 + 2 * b * x ** 2 + x ** 4)
    k_3 = a ** 2 * a_conj ** 2
    k_12 = k_1 * k_2

    if abs(a_conj) >= 0.00000001:
        ret = (1 / (8 * (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** (5 / 2))) * (
            (a + a_conj)
            * (
                (
                    np.sqrt(1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2)
                    * (
                        (-(a ** 5)) * a_conj ** 5 * (1 + x ** 2)
                        - a ** 4 * a_conj ** 4 * (1 + 3 * b - x ** 2) * (1 + x ** 2)
                        - (-1 + b) ** 2
                        * (1 + b)
                        * (1 + x ** 2)
                        * (-(b ** 2) - b * x ** 2 + k_12)
                        + a ** 3
                        * a_conj ** 3
                        * (
                            -2 * b ** 2 * (1 + x ** 2)
                            + 4 * b * (-5 - 2 * x ** 2 + x ** 4)
                            + (1 + x ** 2) * (1 + x ** 2 + k_12)
                        )
                        + k_3
                        * (
                            2 * b ** 3 * (1 + x ** 2)
                            + 2 * b ** 2 * (-19 - 8 * x ** 2 + 3 * x ** 4)
                            + (1 + x ** 2) * (1 - x ** 2 + k_12)
                            + b * (1 + x ** 4 + k_12 + x ** 2 * (18 + k_12))
                        )
                        - k_0
                        * (
                            -3 * b ** 4 * (1 + x ** 2)
                            - 4 * b ** 3 * (-5 - 2 * x ** 2 + x ** 4)
                            + (1 + x ** 2) * (x ** 2 + k_12)
                            + b ** 2 * (1 + x ** 4 + k_12 + x ** 2 * (18 + k_12))
                            - 2
                            * b
                            * (1 - 9 * x ** 4 + 9 * k_12 + x ** 2 * (-4 + 9 * k_12))
                        )
                    )
                )
                / (k_0 * b * (1 + x ** 2) * k_2)
                - 12
                * (-1 + (k_0 + b) ** 2)
                * np.arctanh(
                    (k_3 + (-1 + b) * b + a * (a_conj + 2 * a_conj * b))
                    / (k_1 * np.sqrt(k_3 + (-1 + b) ** 2 + 2 * k_0 * (1 + b)))
                )
                + 12
                * (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2)
                * np.arctanh(
                    (
                        k_3
                        + b ** 2
                        - x ** 2
                        + k_0 * (1 + 2 * b - x ** 2)
                        + b * (-1 + x ** 2)
                    )
                    / (
                        np.sqrt(k_3 + (-1 + b) ** 2 + 2 * k_0 * (1 + b))
                        * np.sqrt((k_0 + b) ** 2 - 2 * (k_0 - b) * x ** 2 + x ** 4)
                    )
                )
                - 24
                * (1 + k_0 - b)
                * np.arctanh(
                    (
                        k_3
                        + b ** 2
                        - x ** 2
                        + k_0 * (1 + 2 * b - x ** 2)
                        + b * (-1 + x ** 2)
                    )
                    / (
                        np.sqrt(k_3 + (-1 + b) ** 2 + 2 * k_0 * (1 + b))
                        * np.sqrt(k_3 + 2 * k_0 * (b - x ** 2) + (b + x ** 2) ** 2)
                    )
                )
            )
        )

        return -8 * np.pi * c * ret

    else:

        return 0


def xxx_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return:
    """
    a_conj = np.conj(a)

    # helper variables to tidy up function
    k_0 = a * a_conj
    k_1 = np.sqrt((k_0 + b) ** 2)
    k_2 = a * a_conj ** 3
    k_3 = a ** 2 * a_conj ** 2
    k_4 = a * a_conj ** 4

    if abs(a_conj) >= 0.00000001:
        ret = (1 / (4 * a ** 3 * a_conj ** 3)) * (
            -(a ** 3)
            + a ** 4 * a_conj
            - a_conj ** 3
            + k_4
            + 2 * (a ** 3 + a_conj ** 3)
            + a ** 3 * b
            + a_conj ** 3 * b
            - (1 / (b * (k_3 + (-1 + b) ** 2 + 2 * k_0 * (1 + b)) ** 3))
            * (
                k_1
                * (
                    a ** 9 * a_conj ** 6 * b
                    + a_conj ** 3 * (-1 + b) ** 5 * b ** 2
                    + k_4 * (-1 + b) ** 3 * b * (2 + 5 * b + 6 * b ** 2)
                    + a ** 8 * a_conj ** 5 * (-1 + 7 * b + 6 * b ** 2)
                    + a ** 7
                    * a_conj ** 4
                    * (-3 - 3 * a_conj ** 2 + 22 * b + 23 * b ** 2 + 15 * b ** 3)
                    + a ** 6
                    * a_conj ** 3
                    * (
                        -3
                        - 3 * a_conj ** 4
                        - b
                        + a_conj ** 6 * b
                        + 32 * b ** 2
                        + 22 * b ** 3
                        + 20 * b ** 4
                        + a_conj ** 2 * (-9 + 30 * b)
                    )
                    + a ** 5
                    * a_conj ** 2
                    * (
                        -1
                        - 19 * b
                        + 3 * b ** 2
                        + 4 * b ** 3
                        - 2 * b ** 4
                        + 15 * b ** 5
                        + a_conj ** 4 * (-9 + 30 * b)
                        + a_conj ** 6 * (-1 + 7 * b + 6 * b ** 2)
                        + 3 * a_conj ** 2 * (-3 - 5 * b + 22 * b ** 2)
                    )
                    + a ** 4
                    * (
                        a_conj * (-1 + b) ** 3 * b * (2 + 5 * b + 6 * b ** 2)
                        + 3 * a_conj ** 5 * (-3 - 5 * b + 22 * b ** 2)
                        + 3 * a_conj ** 3 * (-1 - 14 * b + 5 * b ** 2 + 10 * b ** 3)
                        + a_conj ** 7 * (-3 + 22 * b + 23 * b ** 2 + 15 * b ** 3)
                    )
                    + a ** 2
                    * a_conj ** 3
                    * (-1 + b)
                    * (
                        -3 * (-1 + b) ** 2 * b
                        + a_conj ** 2
                        * (1 + 20 * b + 17 * b ** 2 + 13 * b ** 3 + 15 * b ** 4)
                    )
                    + a ** 3
                    * (
                        -3 * a_conj ** 2 * (-1 + b) ** 3 * b
                        + (-1 + b) ** 5 * b ** 2
                        + 3 * a_conj ** 4 * (-1 - 14 * b + 5 * b ** 2 + 10 * b ** 3)
                        + a_conj ** 6
                        * (-3 - b + 32 * b ** 2 + 22 * b ** 3 + 20 * b ** 4)
                    )
                )
            )
            - ((a ** 3 + a_conj ** 3) * (-1 + k_0 + b)) / (1 + x ** 2) ** 2
            - (2 * (a ** 3 + a_conj ** 3)) / (1 + x ** 2)
            + (1 / (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** 3)
            * (
                np.sqrt(-4 * k_0 * x ** 2 + (k_0 + b + x ** 2) ** 2)
                * (
                    (1 / (1 + x ** 2) ** 2)
                    * (
                        (a + a_conj)
                        * (k_3 + (-1 + b) ** 2 + 2 * k_0 * (1 + b))
                        * (
                            a ** 6 * a_conj ** 4
                            + a_conj ** 2 * (-1 + b) ** 4
                            + a ** 5 * a_conj ** 3 * (2 - a_conj ** 2 + 4 * b)
                            + a ** 4
                            * (
                                a_conj ** 6
                                + 6 * a_conj ** 2 * b ** 2
                                - 2 * a_conj ** 4 * (1 + 2 * b)
                            )
                            + a ** 2
                            * (
                                (-1 + b) ** 4
                                + 6 * a_conj ** 4 * b ** 2
                                - 2 * a_conj ** 2 * (-1 + b) ** 2 * (1 + 2 * b)
                            )
                            + a
                            * (
                                (-a_conj) * (-1 + b) ** 4
                                + a_conj ** 3 * (-1 + b) ** 2 * (2 + 4 * b)
                            )
                            + 2
                            * a ** 3
                            * (
                                a_conj ** 5 * (1 + 2 * b)
                                + a_conj * (-1 + b) ** 2 * (1 + 2 * b)
                                - 3 * a_conj ** 3 * (1 + b ** 2)
                            )
                        )
                    )
                    + (1 / (1 + x ** 2))
                    * (
                        3 * a ** 8 * a_conj ** 5
                        + a_conj ** 3 * (-1 + b) ** 5
                        + 13 * a ** 7 * a_conj ** 4 * (1 + b)
                        + 7 * k_4 * (-1 + b) ** 3 * (1 + b)
                        + 2
                        * a ** 6
                        * a_conj ** 3
                        * (-1 + 12 * a_conj ** 2 + 11 * b + 11 * b ** 2)
                        + 18 * a ** 2 * a_conj ** 5 * (-1 + b ** 3)
                        + a ** 4
                        * (
                            13 * a_conj ** 7 * (1 + b)
                            + 7 * a_conj * (-1 + b) ** 3 * (1 + b)
                            + 6 * a_conj ** 5 * (1 + 8 * b)
                            + 6 * a_conj ** 3 * (-3 - b + 4 * b ** 2)
                        )
                        + a ** 3
                        * (
                            (-1 + b) ** 5
                            + 6 * a_conj ** 4 * (-3 - b + 4 * b ** 2)
                            + a_conj ** 6 * (-2 + 22 * b + 22 * b ** 2)
                        )
                        + 3
                        * a ** 5
                        * (
                            8 * a_conj ** 6
                            + a_conj ** 8
                            + 2 * a_conj ** 4 * (1 + 8 * b)
                            + 6 * a_conj ** 2 * (-1 + b ** 3)
                        )
                    )
                    - (1 / (b * (k_3 + 2 * k_0 * (b - x ** 2) + (b + x ** 2) ** 2)))
                    * (
                        k_0
                        * (a + a_conj) ** 3
                        * (
                            a ** 6 * a_conj ** 6
                            + (-1 + b) ** 3 * b ** 2 * (b + x ** 2)
                            - a ** 5 * a_conj ** 5 * (-3 + 2 * b + x ** 2)
                            + a ** 4
                            * a_conj ** 4
                            * (3 - 17 * b ** 2 - 3 * x ** 2 - 3 * b * (-3 + x ** 2))
                            + a ** 3
                            * a_conj ** 3
                            * (
                                1
                                - 28 * b ** 3
                                - 3 * x ** 2
                                - 2 * b ** 2 * (-3 + x ** 2)
                                - 12 * b * (-1 + x ** 2)
                            )
                            - k_3
                            * (
                                17 * b ** 4
                                + x ** 2
                                - 2 * b ** 3 * (-3 + x ** 2)
                                + 18 * b ** 2 * (-1 + x ** 2)
                                + b * (-1 + 3 * x ** 2)
                            )
                            - k_0
                            * (-1 + b)
                            * b
                            * (
                                2 * b ** 3
                                + 6 * x ** 2
                                + b ** 2 * (11 - 3 * x ** 2)
                                + b * (-1 + 9 * x ** 2)
                            )
                        )
                    )
                )
            )
            + (1 / (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** (7 / 2))
            * (
                12
                * a ** 3
                * a_conj ** 3
                * (a + a_conj)
                * (
                    a ** 4 * a_conj ** 2 * (-2 + a_conj ** 2)
                    - 4 * k_2 * (1 + b)
                    + 4 * a ** 3 * a_conj * (-1 + (-1 + a_conj ** 2) * b)
                    + 4 * k_0 * b * (-1 + b + b ** 2)
                    - 2 * a_conj ** 2 * (1 + 3 * b + b ** 2)
                    + (-1 + b) ** 2 * (1 + 4 * b + b ** 2)
                    - 2
                    * a ** 2
                    * (
                        1
                        + a_conj ** 4
                        + 3 * b
                        + b ** 2
                        - a_conj ** 2 * (-1 + b + 3 * b ** 2)
                    )
                )
                * np.log(
                    k_0
                    - b
                    + (k_0 + b) ** 2
                    + k_1 * np.sqrt(1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2)
                )
            )
            + (
                12
                * a ** 3
                * a_conj ** 3
                * (a + a_conj)
                * (
                    a ** 4 * a_conj ** 2 * (-2 + a_conj ** 2)
                    - 4 * k_2 * (1 + b)
                    + 4 * a ** 3 * a_conj * (-1 + (-1 + a_conj ** 2) * b)
                    + 4 * k_0 * b * (-1 + b + b ** 2)
                    - 2 * a_conj ** 2 * (1 + 3 * b + b ** 2)
                    + (-1 + b) ** 2 * (1 + 4 * b + b ** 2)
                    - 2
                    * a ** 2
                    * (
                        1
                        + a_conj ** 4
                        + 3 * b
                        + b ** 2
                        - a_conj ** 2 * (-1 + b + 3 * b ** 2)
                    )
                )
                * np.log(1 + x ** 2)
            )
            / (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** (7 / 2)
            - (1 / (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** (7 / 2))
            * (
                12
                * a ** 3
                * a_conj ** 3
                * (a + a_conj)
                * (
                    a ** 4 * a_conj ** 2 * (-2 + a_conj ** 2)
                    - 4 * k_2 * (1 + b)
                    + 4 * a ** 3 * a_conj * (-1 + (-1 + a_conj ** 2) * b)
                    + 4 * k_0 * b * (-1 + b + b ** 2)
                    - 2 * a_conj ** 2 * (1 + 3 * b + b ** 2)
                    + (-1 + b) ** 2 * (1 + 4 * b + b ** 2)
                    - 2
                    * a ** 2
                    * (
                        1
                        + a_conj ** 4
                        + 3 * b
                        + b ** 2
                        - a_conj ** 2 * (-1 + b + 3 * b ** 2)
                    )
                )
                * np.log(
                    k_0
                    - b
                    + (k_0 + b) ** 2
                    - x ** 2
                    - k_0 * x ** 2
                    + b * x ** 2
                    + np.sqrt(1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2)
                    * np.sqrt(-4 * k_0 * x ** 2 + (k_0 + b + x ** 2) ** 2)
                )
            )
        )

        return 4 * np.pi * c * ret

    else:

        return 0


def xxy_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return:
    """
    a_conj = np.conj(a)

    # helper variables to tidy up function
    k_0 = a * a_conj
    k_1 = np.sqrt((k_0 + b) ** 2)
    k_2 = a ** 3 * a_conj ** 3
    k_3 = a ** 2 * a_conj ** 2

    if abs(a_conj) >= 0.00000001:
        ret = (
            (1 / 4)
            * (a - a_conj)
            * (
                (2 * (a ** 2 + k_0 + a_conj ** 2)) / k_2
                + ((a ** 2 + k_0 + a_conj ** 2) * (-1 + k_0 + b)) / k_2
                - (1 / (k_2 * b * (k_3 + (-1 + b) ** 2 + 2 * k_0 * (1 + b)) ** 3))
                * (
                    k_1
                    * (
                        a ** 8 * a_conj ** 6 * b
                        + a_conj ** 2 * (-1 + b) ** 5 * b ** 2
                        + a ** 7
                        * a_conj ** 5
                        * (-1 + (7 + a_conj ** 2) * b + 6 * b ** 2)
                        + k_0
                        * b
                        * (
                            (-1 + b) ** 5 * b
                            + a_conj ** 2 * (-1 + b) ** 3 * (2 + 5 * b + 6 * b ** 2)
                        )
                        + a ** 6
                        * a_conj ** 4
                        * (
                            -3
                            + 22 * b
                            + a_conj ** 4 * b
                            + 23 * b ** 2
                            + 15 * b ** 3
                            + a_conj ** 2 * (-2 + 7 * b + 6 * b ** 2)
                        )
                        + a ** 5
                        * a_conj ** 3
                        * (
                            -3
                            - b
                            + 32 * b ** 2
                            + 22 * b ** 3
                            + 20 * b ** 4
                            + a_conj ** 4 * (-1 + 7 * b + 6 * b ** 2)
                            + a_conj ** 2 * (-6 + 32 * b + 23 * b ** 2 + 15 * b ** 3)
                        )
                        + a ** 2
                        * (-1 + b)
                        * (
                            (-1 + b) ** 4 * b ** 2
                            + a_conj ** 2 * (-1 + b) ** 2 * b * (1 + 5 * b + 6 * b ** 2)
                            + a_conj ** 4
                            * (1 + 20 * b + 17 * b ** 2 + 13 * b ** 3 + 15 * b ** 4)
                        )
                        + a ** 4
                        * a_conj ** 2
                        * (
                            -1
                            - 19 * b
                            + 3 * b ** 2
                            + 4 * b ** 3
                            - 2 * b ** 4
                            + 15 * b ** 5
                            + a_conj ** 4 * (-3 + 22 * b + 23 * b ** 2 + 15 * b ** 3)
                            + a_conj ** 2
                            * (-6 - 6 * b + 54 * b ** 2 + 22 * b ** 3 + 20 * b ** 4)
                        )
                        + a ** 3
                        * (
                            a_conj * (-1 + b) ** 3 * b * (2 + 5 * b + 6 * b ** 2)
                            + a_conj ** 5
                            * (-3 - b + 32 * b ** 2 + 22 * b ** 3 + 20 * b ** 4)
                            + a_conj ** 3
                            * (
                                -2
                                - 33 * b
                                + 8 * b ** 2
                                + 14 * b ** 3
                                - 2 * b ** 4
                                + 15 * b ** 5
                            )
                        )
                    )
                )
                - ((a ** 2 + k_0 + a_conj ** 2) * (-1 + k_0 + b))
                / (k_2 * (1 + x ** 2) ** 2)
                - (2 * (a ** 2 + k_0 + a_conj ** 2)) / (k_2 * (1 + x ** 2))
                + (1 / (k_2 * (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** 3))
                * (
                    np.sqrt((k_0 + b) ** 2 - 2 * k_0 * x ** 2 + 2 * b * x ** 2 + x ** 4)
                    * (
                        (1 / (1 + x ** 2) ** 2)
                        * (
                            (k_3 + (-1 + b) ** 2 + 2 * k_0 * (1 + b))
                            * (
                                a ** 6 * a_conj ** 4
                                + a_conj ** 2 * (-1 + b) ** 4
                                + a ** 5 * a_conj ** 3 * (2 + a_conj ** 2 + 4 * b)
                                + a ** 2
                                * (
                                    (-1 + b) ** 4
                                    + 6 * a_conj ** 4 * b ** 2
                                    + 2 * a_conj ** 2 * (-1 + b) ** 2 * (1 + 2 * b)
                                )
                                + a ** 4
                                * (
                                    a_conj ** 6
                                    + 6 * a_conj ** 2 * b ** 2
                                    + a_conj ** 4 * (2 + 4 * b)
                                )
                                + k_0
                                * (
                                    (-1 + b) ** 4
                                    + a_conj ** 2 * (-1 + b) ** 2 * (2 + 4 * b)
                                )
                                + 2
                                * a ** 3
                                * (
                                    a_conj ** 5 * (1 + 2 * b)
                                    + a_conj * (-1 + b) ** 2 * (1 + 2 * b)
                                    + a_conj ** 3 * (-1 + 3 * b ** 2)
                                )
                            )
                        )
                        + (1 / (1 + x ** 2))
                        * (
                            3 * a ** 7 * a_conj ** 5
                            + a_conj ** 2 * (-1 + b) ** 5
                            + a ** 6 * (3 * a_conj ** 6 + 13 * a_conj ** 4 * (1 + b))
                            + k_0
                            * (
                                (-1 + b) ** 5
                                + 7 * a_conj ** 2 * (-1 + b) ** 3 * (1 + b)
                            )
                            + a ** 5
                            * a_conj ** 3
                            * (
                                -2
                                + 3 * a_conj ** 4
                                + 22 * b
                                + 22 * b ** 2
                                + a_conj ** 2 * (21 + 13 * b)
                            )
                            + a ** 4
                            * (
                                13 * a_conj ** 6 * (1 + b)
                                + 2 * a_conj ** 4 * b * (19 + 11 * b)
                                + 18 * a_conj ** 2 * (-1 + b ** 3)
                            )
                            + a ** 2
                            * (
                                (-1 + b) ** 5
                                + 7 * a_conj ** 2 * (-1 + b) ** 3 * (1 + b)
                                + 18 * a_conj ** 4 * (-1 + b ** 3)
                            )
                            + a ** 3
                            * (
                                7 * a_conj * (-1 + b) ** 3 * (1 + b)
                                + a_conj ** 5 * (-2 + 22 * b + 22 * b ** 2)
                                + 2 * a_conj ** 3 * (-12 - b + 4 * b ** 2 + 9 * b ** 3)
                            )
                        )
                        - (1 / (b * (k_3 + 2 * k_0 * (b - x ** 2) + (b + x ** 2) ** 2)))
                        * (
                            k_0
                            * (a + a_conj) ** 2
                            * (
                                a ** 6 * a_conj ** 6
                                + (-1 + b) ** 3 * b ** 2 * (b + x ** 2)
                                - a ** 5 * a_conj ** 5 * (-3 + 2 * b + x ** 2)
                                + a ** 4
                                * a_conj ** 4
                                * (3 - 17 * b ** 2 - 3 * x ** 2 - 3 * b * (-3 + x ** 2))
                                + k_2
                                * (
                                    1
                                    - 28 * b ** 3
                                    - 3 * x ** 2
                                    - 2 * b ** 2 * (-3 + x ** 2)
                                    - 12 * b * (-1 + x ** 2)
                                )
                                - k_3
                                * (
                                    17 * b ** 4
                                    + x ** 2
                                    - 2 * b ** 3 * (-3 + x ** 2)
                                    + 18 * b ** 2 * (-1 + x ** 2)
                                    + b * (-1 + 3 * x ** 2)
                                )
                                - k_0
                                * (-1 + b)
                                * b
                                * (
                                    2 * b ** 3
                                    + 6 * x ** 2
                                    + b ** 2 * (11 - 3 * x ** 2)
                                    + b * (-1 + 9 * x ** 2)
                                )
                            )
                        )
                    )
                )
                + (1 / (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** (7 / 2))
                * (
                    4
                    * (
                        a ** 4 * a_conj ** 2 * (-6 + a_conj ** 2)
                        - 6 * a_conj ** 2 * (1 + 3 * b + b ** 2)
                        + (-1 + b) ** 2 * (1 + 4 * b + b ** 2)
                        + 4 * a ** 3 * (a_conj ** 3 * (-2 + b) - 3 * a_conj * (1 + b))
                        - 4
                        * k_0
                        * (2 + 7 * b + b ** 2 - b ** 3 + 3 * a_conj ** 2 * (1 + b))
                        - 2
                        * a ** 2
                        * (
                            3 * a_conj ** 4
                            + a_conj ** 2 * (9 + 7 * b - 3 * b ** 2)
                            + 3 * (1 + 3 * b + b ** 2)
                        )
                    )
                    * np.log(
                        k_0
                        - b
                        + (k_0 + b) ** 2
                        + k_1 * np.sqrt(1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2)
                    )
                )
                + (
                    4
                    * (
                        a ** 4 * a_conj ** 2 * (-6 + a_conj ** 2)
                        - 6 * a_conj ** 2 * (1 + 3 * b + b ** 2)
                        + (-1 + b) ** 2 * (1 + 4 * b + b ** 2)
                        + 4 * a ** 3 * (a_conj ** 3 * (-2 + b) - 3 * a_conj * (1 + b))
                        - 4
                        * k_0
                        * (2 + 7 * b + b ** 2 - b ** 3 + 3 * a_conj ** 2 * (1 + b))
                        - 2
                        * a ** 2
                        * (
                            3 * a_conj ** 4
                            + a_conj ** 2 * (9 + 7 * b - 3 * b ** 2)
                            + 3 * (1 + 3 * b + b ** 2)
                        )
                    )
                    * np.log(1 + x ** 2)
                )
                / (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** (7 / 2)
                - (1 / (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** (7 / 2))
                * (
                    4
                    * (
                        a ** 4 * a_conj ** 2 * (-6 + a_conj ** 2)
                        - 6 * a_conj ** 2 * (1 + 3 * b + b ** 2)
                        + (-1 + b) ** 2 * (1 + 4 * b + b ** 2)
                        + 4 * a ** 3 * (a_conj ** 3 * (-2 + b) - 3 * a_conj * (1 + b))
                        - 4
                        * k_0
                        * (2 + 7 * b + b ** 2 - b ** 3 + 3 * a_conj ** 2 * (1 + b))
                        - 2
                        * a ** 2
                        * (
                            3 * a_conj ** 4
                            + a_conj ** 2 * (9 + 7 * b - 3 * b ** 2)
                            + 3 * (1 + 3 * b + b ** 2)
                        )
                    )
                    * np.log(
                        k_0
                        - b
                        + (k_0 + b) ** 2
                        - x ** 2
                        - k_0 * x ** 2
                        + b * x ** 2
                        + np.sqrt(1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2)
                        * np.sqrt(k_3 + 2 * k_0 * (b - x ** 2) + (b + x ** 2) ** 2)
                    )
                )
            )
        )

        return (4 / 1j) * np.pi * c * ret

    else:

        return 0


def xxz_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return:
    """
    a_conj = np.conj(a)

    # helper variables to tidy up function
    k_0 = a * a_conj
    k_1 = a * a_conj ** 3
    k_2 = a ** 2 * a_conj ** 2
    k_3 = a ** 4 * a_conj ** 2

    if abs(a_conj) >= 0.00000001:
        ret = (1 / 2) * (
            (1 / (k_2 * b * (k_2 + (-1 + b) ** 2 + 2 * k_0 * (1 + b)) ** 3))
            * (
                np.sqrt((k_0 + b) ** 2)
                * (
                    (-(a ** 7)) * a_conj ** 5
                    + a_conj ** 2 * (-1 + b) ** 4 * b
                    - a ** 6 * a_conj ** 4 * (2 + 2 * a_conj ** 2 + b)
                    - a ** 5
                    * (
                        a_conj ** 7
                        + a_conj ** 5 * (4 - 8 * b)
                        + 2 * a_conj ** 3 * (23 - 2 * b) * b
                    )
                    + k_1 * (-1 + b) ** 2 * (1 + 12 * b + 5 * b ** 2)
                    - k_3
                    * (
                        -2
                        + 36 * b
                        + 4 * a_conj ** 2 * (16 - 9 * b) * b
                        + 38 * b ** 2
                        - 8 * b ** 3
                        + a_conj ** 4 * (2 + b)
                    )
                    + a ** 2
                    * (
                        (-1 + b) ** 4 * b
                        + 2 * a_conj ** 2 * (-1 + b) ** 2 * (1 + 14 * b + 7 * b ** 2)
                        + a_conj ** 4 * (2 - 36 * b - 38 * b ** 2 + 8 * b ** 3)
                    )
                    + a ** 3
                    * (
                        2 * a_conj ** 5 * b * (-23 + 2 * b)
                        + a_conj * (-1 + b) ** 2 * (1 + 12 * b + 5 * b ** 2)
                        + a_conj ** 3 * (4 - 48 * b - 60 * b ** 2 + 40 * b ** 3)
                    )
                )
            )
            - (a ** 2 + a_conj ** 2) / (k_2 * (1 + x ** 2) ** 2)
            + (a ** 2 + a_conj ** 2) / (k_2 * (1 + x ** 2))
            + (1 / (k_2 * (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** 3))
            * (
                np.sqrt((k_0 + b) ** 2 - 2 * k_0 * x ** 2 + 2 * b * x ** 2 + x ** 4)
                * (
                    (
                        (k_2 + (-1 + b) ** 2 + 2 * k_0 * (1 + b))
                        * (
                            a ** 5 * a_conj ** 3
                            + a_conj ** 2 * (-1 + b) ** 3
                            + 3 * k_3 * (1 + b)
                            + 3 * k_1 * (-1 + b ** 2)
                            + a ** 3
                            * a_conj
                            * (-3 + 4 * a_conj ** 2 + a_conj ** 4 + 3 * b ** 2)
                            + a ** 2
                            * (
                                4 * a_conj ** 2 * (-1 + b)
                                + (-1 + b) ** 3
                                + 3 * a_conj ** 4 * (1 + b)
                            )
                        )
                    )
                    / (1 + x ** 2) ** 2
                    - (1 / (1 + x ** 2))
                    * (
                        a ** 7 * a_conj ** 5
                        + a_conj ** 2 * (-1 + b) ** 4 * b
                        + a ** 6 * a_conj ** 4 * (4 + 5 * b)
                        + a ** 5
                        * a_conj ** 3
                        * (-24 + 12 * a_conj ** 2 + a_conj ** 4 + 8 * b + 10 * b ** 2)
                        + k_1 * (-1 + 4 * b - 8 * b ** 3 + 5 * b ** 4)
                        + k_3
                        * (
                            -28
                            - 30 * b
                            + 10 * b ** 3
                            + 12 * a_conj ** 2 * (-2 + 3 * b)
                            + a_conj ** 4 * (4 + 5 * b)
                        )
                        + a ** 3
                        * a_conj
                        * (
                            -1
                            + 4 * b
                            - 8 * b ** 3
                            + 5 * b ** 4
                            + 2 * a_conj ** 4 * (-12 + 4 * b + 5 * b ** 2)
                            + 4 * a_conj ** 2 * (-7 - 10 * b + 9 * b ** 2)
                        )
                        + a ** 2
                        * (
                            (-1 + b) ** 4 * b
                            + 4 * a_conj ** 2 * (-1 + b) ** 2 * (2 + 3 * b)
                            + 2 * a_conj ** 4 * (-14 - 15 * b + 5 * b ** 3)
                        )
                    )
                    + (1 / (b * (k_2 + 2 * k_0 * (b - x ** 2) + (b + x ** 2) ** 2)))
                    * (
                        k_0
                        * (a + a_conj) ** 2
                        * (
                            a ** 6 * a_conj ** 6
                            + a ** 5 * a_conj ** 5 * (2 + 2 * b - x ** 2)
                            - (-1 + b) ** 2
                            * b
                            * (
                                b
                                + 3 * b ** 3
                                + 3 * x ** 2
                                + 8 * b * x ** 2
                                + b ** 2 * (8 + x ** 2)
                            )
                            - 2
                            * a ** 3
                            * a_conj ** 3
                            * (
                                1
                                + 10 * b ** 3
                                + 6 * b * (-1 + x ** 2)
                                + b ** 2 * (-26 + 5 * x ** 2)
                            )
                            - a ** 4
                            * a_conj ** 4
                            * (5 * b ** 2 + 2 * x ** 2 + b * (-22 + 5 * x ** 2))
                            - k_2
                            * (
                                1
                                + 25 * b ** 4
                                - 2 * x ** 2
                                + 12 * b ** 2 * (-3 + 2 * x ** 2)
                                + 2 * b ** 3 * (-22 + 5 * x ** 2)
                                - 2 * b * (-5 + 6 * x ** 2)
                            )
                            + k_0
                            * (
                                -14 * b ** 5
                                + x ** 2
                                - 5 * b ** 4 * (-2 + x ** 2)
                                - 4 * b ** 3 * (-9 + 5 * x ** 2)
                                + 2 * b * (-1 + 8 * x ** 2)
                                + 2 * b ** 2 * (-7 + 12 * x ** 2)
                            )
                        )
                    )
                )
            )
            - (1 / (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** (7 / 2))
            * (
                4
                * (-1 + k_0 + b)
                * (
                    k_3 * (-6 + a_conj ** 2)
                    - 6 * a_conj ** 2 * (1 + 3 * b + b ** 2)
                    + (-1 + b) ** 2 * (1 + 4 * b + b ** 2)
                    + 4 * a ** 3 * (a_conj ** 3 * (-2 + b) - 3 * a_conj * (1 + b))
                    - 4
                    * k_0
                    * (2 + 7 * b + b ** 2 - b ** 3 + 3 * a_conj ** 2 * (1 + b))
                    - 2
                    * a ** 2
                    * (
                        3 * a_conj ** 4
                        + a_conj ** 2 * (9 + 7 * b - 3 * b ** 2)
                        + 3 * (1 + 3 * b + b ** 2)
                    )
                )
                * np.arctanh(
                    (k_2 + (-1 + b) * b + a * (a_conj + 2 * a_conj * b))
                    / (
                        np.sqrt((k_0 + b) ** 2)
                        * np.sqrt(k_2 + (-1 + b) ** 2 + 2 * k_0 * (1 + b))
                    )
                )
            )
            + (1 / (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** (7 / 2))
            * (
                4
                * (-1 + k_0 + b)
                * (
                    k_3 * (-6 + a_conj ** 2)
                    - 6 * a_conj ** 2 * (1 + 3 * b + b ** 2)
                    + (-1 + b) ** 2 * (1 + 4 * b + b ** 2)
                    + 4 * a ** 3 * (a_conj ** 3 * (-2 + b) - 3 * a_conj * (1 + b))
                    - 4
                    * k_0
                    * (2 + 7 * b + b ** 2 - b ** 3 + 3 * a_conj ** 2 * (1 + b))
                    - 2
                    * a ** 2
                    * (
                        3 * a_conj ** 4
                        + a_conj ** 2 * (9 + 7 * b - 3 * b ** 2)
                        + 3 * (1 + 3 * b + b ** 2)
                    )
                )
                * np.arctanh(
                    (k_2 + k_0 * (1 + 2 * b - x ** 2) + (-1 + b) * (b + x ** 2))
                    / (
                        np.sqrt(k_2 + (-1 + b) ** 2 + 2 * k_0 * (1 + b))
                        * np.sqrt(k_2 + 2 * k_0 * (b - x ** 2) + (b + x ** 2) ** 2)
                    )
                )
            )
        )

        return 2 * np.pi * c * ret

    if abs(a_conj) < 0.00000001 and (b - 1) ** 2 > 0.00000001:
        ret = (
            -1
            - (-1 + b) ** 2
            + b ** 2
            + (-1 + b) * (1 + 3 * b)
            + (-1 + b) ** 2 / (1 + x ** 2) ** 2
            + (1 + 2 * b - 3 * b ** 2) / (1 + x ** 2)
            + (b - b ** 3) / (b + x ** 2)
            - (1 + 4 * b + b ** 2) * np.log(b)
            - (1 + 4 * b + b ** 2) * np.log(1 + x ** 2)
            + (1 + 4 * b + b ** 2) * np.log(b + x ** 2)
        ) / (-1 + b) ** 4

        return 4 * np.pi * c * ret

    if abs(a_conj) < 0.00000001 and (b - 1) ** 2 <= 0.00000001:
        ret = 2 * x ** 4 / (4 * (1 + x ** 2) ** 4)

        return 4 * np.pi * c * ret


def xyz_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return:
    """
    a_conj = np.conj(a)

    # helper variables to tidy up function
    k_0 = a * a_conj
    k_1 = a ** 2 * a_conj ** 2

    if abs(a_conj) >= 0.00000001:
        ret = (
            (1 / 4)
            * (a ** 2 - a_conj ** 2)
            * (
                (
                    np.sqrt((k_0 + b) ** 2)
                    * (
                        (-(a ** 5)) * a_conj ** 5
                        + (-1 + b) ** 4 * b
                        - a ** 4 * a_conj ** 4 * (2 + b)
                        + 2 * a ** 3 * a_conj ** 3 * b * (-23 + 2 * b)
                        + k_0 * (-1 + b) ** 2 * (1 + 12 * b + 5 * b ** 2)
                        + 2 * k_1 * (1 - 18 * b - 19 * b ** 2 + 4 * b ** 3)
                    )
                )
                / (k_1 * b * (k_1 + (-1 + b) ** 2 + 2 * k_0 * (1 + b)) ** 3)
                - 1 / (k_1 * (1 + x ** 2) ** 2)
                + 1 / (k_1 * (1 + x ** 2))
                - (
                    (-1 + b) ** 4 * b * (b + x ** 2) ** 2 * (1 + b * x ** 2)
                    - a ** 7 * a_conj ** 7 * (1 - (-2 + b) * x ** 2 + x ** 4)
                    + a ** 6
                    * a_conj ** 6
                    * (
                        -2
                        - 3 * x ** 2
                        + 7 * b ** 2 * x ** 2
                        + x ** 6
                        - b * (3 + 4 * x ** 4)
                    )
                    + a ** 5
                    * a_conj ** 5
                    * (
                        21 * b ** 3 * x ** 2
                        + 2 * (x + x ** 3) ** 2
                        + b ** 2 * (1 + 26 * x ** 2 - 3 * x ** 4)
                        + b * (-50 - 61 * x ** 2 - 20 * x ** 4 + 6 * x ** 6)
                    )
                    + k_0
                    * (-1 + b) ** 2
                    * b
                    * (
                        x ** 2
                        + 7 * b ** 4 * x ** 2
                        + 10 * x ** 4
                        + 2 * x ** 6
                        + b ** 3 * (7 + 4 * x ** 2 + 11 * x ** 4)
                        + 2 * b ** 2 * (4 + 10 * x ** 2 + 9 * x ** 4 + 3 * x ** 6)
                        + b * (3 + 22 * x ** 2 + 15 * x ** 4 + 10 * x ** 6)
                    )
                    + a ** 3
                    * a_conj ** 3
                    * (
                        1
                        + 35 * b ** 5 * x ** 2
                        - 3 * x ** 4
                        - 2 * x ** 6
                        + 25 * b ** 4 * (1 + x ** 2) ** 2
                        + b * (14 + 55 * x ** 2 + 14 * x ** 4 - 36 * x ** 6)
                        + 2 * b ** 3 * (-60 - 83 * x ** 2 - 4 * x ** 4 + 10 * x ** 6)
                        + 2 * b ** 2 * (-45 - 54 * x ** 2 + 11 * x ** 4 + 16 * x ** 6)
                    )
                    + a ** 4
                    * a_conj ** 4
                    * (
                        35 * b ** 4 * x ** 2
                        + 2 * (1 + x ** 2) ** 2
                        + 5 * b ** 3 * (3 + 12 * x ** 2 + 2 * x ** 4)
                        + b ** 2 * (-132 - 170 * x ** 2 - 40 * x ** 4 + 15 * x ** 6)
                        + b * (-36 + 16 * x ** 2 + 59 * x ** 4 + 16 * x ** 6)
                    )
                    + k_1
                    * (
                        21 * b ** 6 * x ** 2
                        - (x + x ** 3) ** 2
                        + b ** 5 * (19 + 8 * x ** 2 + 24 * x ** 4)
                        + 2 * b ** 3 * (-33 - 62 * x ** 2 - 28 * x ** 4 + 10 * x ** 6)
                        + b ** 4 * (-38 - 43 * x ** 2 + 16 * x ** 4 + 15 * x ** 6)
                        - b * (-3 + 20 * x ** 2 + 52 * x ** 4 + 44 * x ** 6)
                        - b ** 2 * (-18 + 33 * x ** 2 + 122 * x ** 4 + 54 * x ** 6)
                    )
                )
                / (
                    k_1
                    * b
                    * (k_1 + (-1 + b) ** 2 + 2 * k_0 * (1 + b)) ** 3
                    * (1 + x ** 2) ** 2
                    * np.sqrt(
                        (k_0 + b) ** 2 - 2 * k_0 * x ** 2 + 2 * b * x ** 2 + x ** 4
                    )
                )
                + (
                    24
                    * (-1 + k_0 + b)
                    * (1 + k_1 + 3 * b + b ** 2 + 2 * k_0 * (1 + b))
                    * np.log(
                        k_0
                        - b
                        + (k_0 + b) ** 2
                        + np.sqrt((k_0 + b) ** 2)
                        * np.sqrt(1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2)
                    )
                )
                / (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** (7 / 2)
                + (
                    24
                    * (-1 + k_0 + b)
                    * (1 + k_1 + 3 * b + b ** 2 + 2 * k_0 * (1 + b))
                    * np.log(1 + x ** 2)
                )
                / (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** (7 / 2)
                - (
                    24
                    * (-1 + k_0 + b)
                    * (1 + k_1 + 3 * b + b ** 2 + 2 * k_0 * (1 + b))
                    * np.log(
                        k_1
                        - b
                        + b ** 2
                        - x ** 2
                        + b * x ** 2
                        + k_0 * (1 + 2 * b - x ** 2)
                        + np.sqrt(k_1 + (-1 + b) ** 2 + 2 * k_0 * (1 + b))
                        * np.sqrt(k_1 + 2 * k_0 * (b - x ** 2) + (b + x ** 2) ** 2)
                    )
                )
                / (1 + 2 * k_0 - 2 * b + (k_0 + b) ** 2) ** (7 / 2)
            )
        )

        return (4 / 1j) * np.pi * c * ret

    else:

        return 0


def zzz_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return:
    """
    a_conj = np.conj(a)

    if abs(a_conj) >= 0.00000001:
        ret = (
            -(
                np.sqrt((a * a_conj + b) ** 2)
                * (
                    a ** 6 * a_conj ** 6
                    + 6 * a ** 5 * a_conj ** 5 * b
                    + 4 * a ** 3 * a_conj ** 3 * b * (-29 + 28 * b + 5 * b ** 2)
                    + a ** 4 * a_conj ** 4 * (-3 + 28 * b + 15 * b ** 2)
                    + (-1 + b) ** 2 * (-1 + 10 * b + 48 * b ** 2 + 30 * b ** 3 + b ** 4)
                    + 2
                    * a
                    * a_conj
                    * b
                    * (39 - 96 * b - 66 * b ** 2 + 56 * b ** 3 + 3 * b ** 4)
                    + 3
                    * a ** 2
                    * a_conj ** 2
                    * (1 - 24 * b - 78 * b ** 2 + 56 * b ** 3 + 5 * b ** 4)
                )
                / (
                    4
                    * b
                    * (a * a_conj + b)
                    * (a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b))
                    ** 3
                )
            )
            + (
                a ** 7 * a_conj ** 7 * (1 + x ** 2) ** 2
                + a ** 6 * a_conj ** 6 * (7 * b - x ** 2) * (1 + x ** 2) ** 2
                + a ** 5
                * a_conj ** 5
                * (
                    -3 * (1 + x ** 2) ** 2
                    + 21 * b ** 2 * (1 + x ** 2) ** 2
                    + b * (28 + 42 * x ** 2 - 6 * x ** 6)
                )
                + a ** 4
                * a_conj ** 4
                * (
                    35 * b ** 3 * (1 + x ** 2) ** 2
                    + 3 * (x + x ** 3) ** 2
                    - 5 * b ** 2 * (-28 - 45 * x ** 2 - 6 * x ** 4 + 3 * x ** 6)
                    + b * (-119 - 194 * x ** 2 - 87 * x ** 4 + 12 * x ** 6)
                )
                + a ** 3
                * a_conj ** 3
                * (
                    3 * (1 + x ** 2) ** 2
                    + 35 * b ** 4 * (1 + x ** 2) ** 2
                    - 20 * b ** 3 * (-14 - 23 * x ** 2 - 4 * x ** 4 + x ** 6)
                    + 4 * b * (-18 + 17 * x ** 2 + 52 * x ** 4 + 21 * x ** 6)
                    + 2 * b ** 2 * (-175 - 270 * x ** 2 - 87 * x ** 4 + 24 * x ** 6)
                )
                + (-1 + b) ** 2
                * (
                    b ** 5 * (1 + x ** 2) ** 2
                    + (x + x ** 3) ** 2
                    + b ** 4 * (30 + 51 * x ** 2 + 12 * x ** 4 - x ** 6)
                    + 2 * b ** 3 * (24 + 53 * x ** 2 + 46 * x ** 4 + 5 * x ** 6)
                    + 2 * b ** 2 * (5 + 46 * x ** 2 + 53 * x ** 4 + 24 * x ** 6)
                    + b * (-1 + 12 * x ** 2 + 51 * x ** 4 + 30 * x ** 6)
                )
                + a
                * a_conj
                * (
                    -((1 + x ** 2) ** 2)
                    + 7 * b ** 6 * (1 + x ** 2) ** 2
                    + b ** 2 * (105 + 2 * x ** 2 - 327 * x ** 4 - 192 * x ** 6)
                    + b ** 5 * (140 + 234 * x ** 2 + 48 * x ** 4 - 6 * x ** 6)
                    + 4 * b ** 3 * (-62 - 99 * x ** 2 - 16 * x ** 4 + 33 * x ** 6)
                    + b ** 4 * (-143 - 158 * x ** 2 + 129 * x ** 4 + 48 * x ** 6)
                    - 2 * b * (-6 + 39 * x ** 2 + 88 * x ** 4 + 55 * x ** 6)
                )
                + a ** 2
                * a_conj ** 2
                * (
                    21 * b ** 5 * (1 + x ** 2) ** 2
                    - 3 * (x + x ** 3) ** 2
                    + b * (81 + 202 * x ** 2 + 65 * x ** 4 - 72 * x ** 6)
                    + 5 * b ** 4 * (56 + 93 * x ** 2 + 18 * x ** 4 - 3 * x ** 6)
                    + 6 * b ** 3 * (-61 - 86 * x ** 2 - 5 * x ** 4 + 12 * x ** 6)
                    + 2 * b ** 2 * (-132 - 131 * x ** 2 + 102 * x ** 4 + 93 * x ** 6)
                )
            )
            / (
                4
                * b
                * (a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b)) ** 3
                * (1 + x ** 2) ** 2
                * np.sqrt(
                    (a * a_conj + b) ** 2
                    - 2 * a * a_conj * x ** 2
                    + 2 * b * x ** 2
                    + x ** 4
                )
            )
            + 3
            * (-1 + a * a_conj + b)
            * (
                a ** 4 * a_conj ** 4
                + 4 * a ** 3 * a_conj ** 3 * (-1 + b)
                + (-1 + b ** 2) ** 2
                + 2 * a ** 2 * a_conj ** 2 * (-5 - 4 * b + 3 * b ** 2)
                + 4 * a * a_conj * (-1 - 5 * b - b ** 2 + b ** 3)
            )
            * np.log(
                a * a_conj
                - b
                + (a * a_conj + b) ** 2
                + np.sqrt((a * a_conj + b) ** 2)
                * np.sqrt(1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2)
            )
            / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (7 / 2)
            + 3
            * (-1 + a * a_conj + b)
            * (
                a ** 4 * a_conj ** 4
                + 4 * a ** 3 * a_conj ** 3 * (-1 + b)
                + (-1 + b ** 2) ** 2
                + 2 * a ** 2 * a_conj ** 2 * (-5 - 4 * b + 3 * b ** 2)
                + 4 * a * a_conj * (-1 - 5 * b - b ** 2 + b ** 3)
            )
            * np.log(1 + x ** 2)
            / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (7 / 2)
            - 3
            * (-1 + a * a_conj + b)
            * (
                a ** 4 * a_conj ** 4
                + 4 * a ** 3 * a_conj ** 3 * (-1 + b)
                + (-1 + b ** 2) ** 2
                + 2 * a ** 2 * a_conj ** 2 * (-5 - 4 * b + 3 * b ** 2)
                + 4 * a * a_conj * (-1 - 5 * b - b ** 2 + b ** 3)
            )
            * np.log(
                a * a_conj
                - b
                + (a * a_conj + b) ** 2
                - x ** 2
                - a * a_conj * x ** 2
                + b * x ** 2
                + np.sqrt(1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2)
                * np.sqrt(
                    a ** 2 * a_conj ** 2
                    + 2 * a * a_conj * (b - x ** 2)
                    + (b + x ** 2) ** 2
                )
            )
            / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (7 / 2)
        )
        return 4 * c * np.pi * ret

    if abs(a_conj) < 0.00000001 and (b - 1) ** 2 >= 0.00000001:
        ret = (
            4 * (-1 + b) ** 2
            - (-1 + b) * (1 + b) ** 3 / b
            - 4 * (-1 + b) * (1 + 3 * b)
            - 4 * (-1 + b) ** 2 / (1 + x ** 2) ** 2
            + 4 * (-1 + b) * (1 + 3 * b) / (1 + x ** 2)
            + (-1 + b) * (1 + b) ** 3 / (b + x ** 2)
            + 6 * (1 + b) ** 2 * np.log(b)
            - 6 * (1 + b) ** 2 * np.log(b + x ** 2)
            + 6 * (1 + b) ** 2 * np.log(1 + x ** 2)
        ) / (2 * (-1 + b) ** 4)
        return 4 * c * np.pi * ret
    if abs(a_conj) < 0.00000001 and (b - 1) ** 2 < 0.00000001:
        ret = x ** 2 * (1 + x ** 4) / (2 * (1 + x ** 2) ** 4)
        return 4 * c * np.pi * ret


def xxxx_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return:
    """
    a_conj = np.conj(a)

    if abs(a_conj) >= 0.00000001:
        ret = (
            1
            / (12 * a ** 4 * a_conj ** 4)
            * (
                3 * a ** 4
                + 2 * a ** 5 * a_conj
                - 4 * a ** 4 * a_conj ** 2
                + 3 * a ** 6 * a_conj ** 2
                + 3 * a_conj ** 4
                + 2 * a ** 2 * a_conj ** 4
                + 2 * a * a_conj ** 5
                + 3 * a ** 2 * a_conj ** 6
                + 9 * (a ** 4 + a_conj ** 4)
                + 9 * a_conj ** 4 * (-1 + b)
                - 6 * a ** 4 * b
                + 6 * a ** 5 * a_conj * b
                - 6 * a_conj ** 4 * b
                + 6 * a * a_conj ** 5 * b
                + 3 * a ** 4 * b ** 2
                + 3 * a_conj ** 4 * b ** 2
                + a ** 4 * (-9 + 6 * a_conj ** 2 + 9 * b)
                - 1
                / (
                    b
                    * (a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b))
                    ** 4
                )
                * (
                    np.sqrt((a * a_conj + b) ** 2)
                    * (
                        3 * a ** 13 * a_conj ** 9 * b
                        + 3 * a_conj ** 4 * (-1 + b) ** 7 * b ** 3
                        + a ** 12 * a_conj ** 8 * b * (26 + 27 * b)
                        + a
                        * a_conj ** 5
                        * (-1 + b) ** 5
                        * b ** 2
                        * (9 + 14 * b + 27 * b ** 2)
                        + a ** 11
                        * a_conj ** 7
                        * (
                            -3
                            + (103 + 2 * a_conj ** 2) * b
                            + 161 * b ** 2
                            + 108 * b ** 3
                        )
                        + a ** 10
                        * a_conj ** 6
                        * (
                            -12
                            + 251 * b
                            + 409 * b ** 2
                            + 399 * b ** 3
                            + 252 * b ** 4
                            + 2 * a_conj ** 2 * (-6 + 8 * b + 7 * b ** 2)
                        )
                        + a ** 9
                        * a_conj ** 5
                        * (
                            -18
                            + 175 * b
                            + 2 * a_conj ** 6 * b
                            + 3 * a_conj ** 8 * b
                            + 601 * b ** 2
                            + 563 * b ** 3
                            + 469 * b ** 4
                            + 378 * b ** 5
                            - 6 * a_conj ** 4 * (3 + 2 * b)
                            + a_conj ** 2 * (-48 + 266 * b + 64 * b ** 2 + 42 * b ** 3)
                        )
                        + a ** 8
                        * a_conj ** 4
                        * (
                            -12
                            - 112 * b
                            + 307 * b ** 2
                            + 407 * b ** 3
                            + 285 * b ** 4
                            + 175 * b ** 5
                            + 378 * b ** 6
                            + a_conj ** 8 * b * (26 + 27 * b)
                            + 2 * a_conj ** 6 * (-6 + 8 * b + 7 * b ** 2)
                            - 6 * a_conj ** 4 * (12 - 55 * b + 10 * b ** 2)
                            + a_conj ** 2
                            * (-72 + 88 * b + 754 * b ** 2 + 80 * b ** 3 + 70 * b ** 4)
                        )
                        + a ** 2
                        * a_conj ** 4
                        * b
                        * (
                            2 * (-3 + b) * (-1 + b) ** 5 * b
                            + a_conj ** 2
                            * (-1 + b) ** 3
                            * (9 + 70 * b + 74 * b ** 2 + 65 * b ** 3 + 108 * b ** 4)
                        )
                        + a ** 7
                        * a_conj ** 3
                        * (
                            -3
                            - 125 * b
                            - 181 * b ** 2
                            + 96 * b ** 3
                            + 65 * b ** 4
                            + 85 * b ** 5
                            - 189 * b ** 6
                            + 252 * b ** 7
                            - 12
                            * a_conj ** 4
                            * (9 - 13 * b - 78 * b ** 2 + 10 * b ** 3)
                            + a_conj ** 6 * (-48 + 266 * b + 64 * b ** 2 + 42 * b ** 3)
                            + a_conj ** 8 * (-3 + 103 * b + 161 * b ** 2 + 108 * b ** 3)
                            + a_conj ** 2
                            * (
                                -48
                                - 538 * b
                                + 736 * b ** 2
                                + 596 * b ** 3
                                + 70 * b ** 5
                            )
                        )
                        + a ** 5
                        * (
                            a_conj * (-1 + b) ** 5 * b ** 2 * (9 + 14 * b + 27 * b ** 2)
                            + 2
                            * a_conj ** 3
                            * (-1 + b) ** 3
                            * b
                            * (-15 - 37 * b - 11 * b ** 2 + 7 * b ** 3)
                            - 6
                            * a_conj ** 5
                            * (
                                3
                                + 60 * b
                                + 128 * b ** 2
                                - 218 * b ** 3
                                + 17 * b ** 4
                                + 10 * b ** 5
                            )
                            + a_conj ** 7
                            * (
                                -48
                                - 538 * b
                                + 736 * b ** 2
                                + 596 * b ** 3
                                + 70 * b ** 5
                            )
                            + a_conj ** 9
                            * (
                                -18
                                + 175 * b
                                + 601 * b ** 2
                                + 563 * b ** 3
                                + 469 * b ** 4
                                + 378 * b ** 5
                            )
                        )
                        + a ** 6
                        * a_conj ** 2
                        * (
                            -12
                            * a_conj ** 4
                            * (6 + 53 * b - 89 * b ** 2 - 54 * b ** 3 + 10 * b ** 4)
                            + a_conj ** 6
                            * (-72 + 88 * b + 754 * b ** 2 + 80 * b ** 3 + 70 * b ** 4)
                            + (-1 + b) ** 3
                            * b
                            * (9 + 70 * b + 74 * b ** 2 + 65 * b ** 3 + 108 * b ** 4)
                            + a_conj ** 8
                            * (
                                -12
                                + 251 * b
                                + 409 * b ** 2
                                + 399 * b ** 3
                                + 252 * b ** 4
                            )
                            + 2
                            * a_conj ** 2
                            * (
                                -6
                                - 172 * b
                                - 251 * b ** 2
                                + 396 * b ** 3
                                + 52 * b ** 4
                                - 40 * b ** 5
                                + 21 * b ** 6
                            )
                        )
                        + a ** 3
                        * a_conj ** 5
                        * (-1 + b)
                        * (
                            2
                            * (-1 + b) ** 2
                            * b
                            * (-15 - 37 * b - 11 * b ** 2 + 7 * b ** 3)
                            + a_conj ** 2
                            * (
                                3
                                + 128 * b
                                + 309 * b ** 2
                                + 213 * b ** 3
                                + 148 * b ** 4
                                + 63 * b ** 5
                                + 252 * b ** 6
                            )
                        )
                        + a ** 4
                        * (
                            2 * a_conj ** 2 * (-3 + b) * (-1 + b) ** 5 * b ** 2
                            + 3 * (-1 + b) ** 7 * b ** 3
                            - 6
                            * a_conj ** 4
                            * (-1 + b) ** 3
                            * b
                            * (15 + 33 * b + 2 * b ** 2)
                            + 2
                            * a_conj ** 6
                            * (
                                -6
                                - 172 * b
                                - 251 * b ** 2
                                + 396 * b ** 3
                                + 52 * b ** 4
                                - 40 * b ** 5
                                + 21 * b ** 6
                            )
                            + a_conj ** 8
                            * (
                                -12
                                - 112 * b
                                + 307 * b ** 2
                                + 407 * b ** 3
                                + 285 * b ** 4
                                + 175 * b ** 5
                                + 378 * b ** 6
                            )
                        )
                    )
                )
                + (
                    -3 * a ** 6 * a_conj ** 2
                    + a ** 2 * (4 * a_conj ** 4 - 3 * a_conj ** 6)
                    + a ** 4 * (4 * a_conj ** 2 - 3 * (-1 + b) ** 2)
                    + a ** 5 * a_conj * (4 - 6 * b)
                    + 2 * a * a_conj ** 5 * (2 - 3 * b)
                    - 3 * a_conj ** 4 * (-1 + b) ** 2
                )
                / (1 + x ** 2) ** 3
                - 3
                * (
                    2 * a ** 5 * a_conj
                    + 2 * a ** 2 * a_conj ** 4
                    + 2 * a * a_conj ** 5
                    + 3 * a_conj ** 4 * (-1 + b)
                    + a ** 4 * (-3 + 2 * a_conj ** 2 + 3 * b)
                )
                / (1 + x ** 2) ** 2
                - 9 * (a ** 4 + a_conj ** 4) / (1 + x ** 2)
                + 1
                / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** 4
                * (
                    np.sqrt(-4 * a * a_conj * x ** 2 + (a * a_conj + b + x ** 2) ** 2)
                    * (
                        1
                        / (1 + x ** 2) ** 3
                        * (
                            (
                                a ** 2 * a_conj ** 2
                                + (-1 + b) ** 2
                                + 2 * a * a_conj * (1 + b)
                            )
                            ** 2
                            * (
                                3 * a ** 9 * a_conj ** 5
                                + 3 * a_conj ** 4 * (-1 + b) ** 5
                                + 5 * a ** 8 * a_conj ** 4 * (1 + 3 * b)
                                + 5 * a * a_conj ** 5 * (-1 + b) ** 3 * (1 + 3 * b)
                                + a ** 7
                                * (-4 * a_conj ** 5 + 30 * a_conj ** 3 * b ** 2)
                                + 6
                                * a ** 3
                                * a_conj ** 5
                                * (2 + (-2 + 5 * a_conj ** 2) * b ** 2)
                                + 2
                                * a ** 2
                                * a_conj ** 4
                                * (-1 + b)
                                * (-2 + 4 * b + (-2 + 15 * a_conj ** 2) * b ** 2)
                                - 6
                                * a ** 6
                                * (
                                    -5 * a_conj ** 2 * (-1 + b) * b ** 2
                                    + 2 * a_conj ** 4 * (1 + b)
                                )
                                + a ** 4
                                * (
                                    -12 * a_conj ** 4 * (-1 + b)
                                    - 4 * a_conj ** 2 * (-1 + b) ** 3
                                    + 3 * (-1 + b) ** 5
                                    - 12 * a_conj ** 6 * (1 + b)
                                    + 5 * a_conj ** 8 * (1 + 3 * b)
                                )
                                + a ** 5
                                * (
                                    -12 * a_conj ** 5
                                    - 4 * a_conj ** 7
                                    + 3 * a_conj ** 9
                                    + 5 * a_conj * (-1 + b) ** 3 * (1 + 3 * b)
                                    - 12 * a_conj ** 3 * (-1 + b ** 2)
                                )
                            )
                        )
                        + 1
                        / (1 + x ** 2) ** 2
                        * (
                            (
                                a ** 2 * a_conj ** 2
                                + (-1 + b) ** 2
                                + 2 * a * a_conj * (1 + b)
                            )
                            * (
                                9 * a ** 10 * a_conj ** 6
                                + 6 * a_conj ** 4 * (-1 + b) ** 6
                                + a * a_conj ** 5 * (-1 + b) ** 4 * (31 + 39 * b)
                                + a ** 9 * a_conj ** 5 * (29 + 6 * a_conj ** 2 + 51 * b)
                                + 2
                                * a ** 8
                                * (
                                    a_conj ** 6 * (13 + 15 * b)
                                    + 5 * a_conj ** 4 * (2 + 7 * b + 12 * b ** 2)
                                )
                                + a ** 2
                                * (
                                    2 * a_conj ** 4 * (-1 + b) ** 4 * (-1 + 3 * b)
                                    + 5
                                    * a_conj ** 6
                                    * (-1 + b) ** 2
                                    * (11 + 16 * b + 21 * b ** 2)
                                )
                                + 2
                                * a ** 7
                                * a_conj ** 3
                                * (
                                    15
                                    + 18 * a_conj ** 4
                                    + 3 * a_conj ** 6
                                    + 75 * b ** 3
                                    + a_conj ** 2 * (-32 + 26 * b + 30 * b ** 2)
                                )
                                + a ** 5
                                * (
                                    a_conj * (-1 + b) ** 4 * (31 + 39 * b)
                                    + a_conj ** 9 * (29 + 51 * b)
                                    + 36 * a_conj ** 5 * (-2 - 3 * b + 3 * b ** 2)
                                    + 2
                                    * a_conj ** 3
                                    * (-1 + b) ** 2
                                    * (-7 + 4 * b + 15 * b ** 2)
                                    + a_conj ** 7 * (-64 + 52 * b + 60 * b ** 2)
                                )
                                + 2
                                * a ** 3
                                * (
                                    a_conj ** 5
                                    * (-1 + b) ** 2
                                    * (-7 + 4 * b + 15 * b ** 2)
                                    + 15 * a_conj ** 7 * (1 + 5 * b ** 3)
                                )
                                + a ** 6
                                * a_conj ** 2
                                * (
                                    9 * a_conj ** 8
                                    + 12 * a_conj ** 4 * (-4 + 9 * b)
                                    + a_conj ** 6 * (26 + 30 * b)
                                    + 5 * (-1 + b) ** 2 * (11 + 16 * b + 21 * b ** 2)
                                    + 12 * a_conj ** 2 * (-8 - 9 * b + 5 * b ** 3)
                                )
                                + 2
                                * a ** 4
                                * (
                                    3 * (-1 + b) ** 6
                                    + a_conj ** 2 * (-1 + b) ** 4 * (-1 + 3 * b)
                                    + 6 * a_conj ** 4 * (-1 + b) ** 2 * (1 + 3 * b)
                                    + 5 * a_conj ** 8 * (2 + 7 * b + 12 * b ** 2)
                                    + 6 * a_conj ** 6 * (-8 - 9 * b + 5 * b ** 3)
                                )
                            )
                        )
                        + 1
                        / (1 + x ** 2)
                        * (
                            18 * a ** 11 * a_conj ** 7
                            + 3 * a_conj ** 4 * (-1 + b) ** 7
                            + 2 * a * a_conj ** 5 * (-1 + b) ** 5 * (13 + 12 * b)
                            + a ** 10 * a_conj ** 6 * (101 + 6 * a_conj ** 2 + 99 * b)
                            - 2
                            * a ** 9
                            * a_conj ** 5
                            * (
                                -29
                                + 18 * a_conj ** 4
                                - 167 * b
                                - 114 * b ** 2
                                - 4 * a_conj ** 2 * (31 + 3 * b)
                            )
                            + a ** 2
                            * a_conj ** 4
                            * (-1 + b) ** 3
                            * (
                                -2 * (-1 + b) ** 2 * (-1 + 3 * b)
                                + a_conj ** 2 * (103 + 154 * b + 93 * b ** 2)
                            )
                            + 2
                            * a ** 3
                            * a_conj ** 5
                            * (-1 + b)
                            * (
                                -4 * (-1 + b) ** 2 * (-2 + b + 3 * b ** 2)
                                + 5
                                * a_conj ** 2
                                * (23 + 29 * b + 29 * b ** 2 + 21 * b ** 3)
                            )
                            + a ** 8
                            * a_conj ** 4
                            * (
                                6 * a_conj ** 6
                                - 36 * a_conj ** 4 * (-6 + 5 * b)
                                + a_conj ** 2 * (322 + 712 * b + 30 * b ** 2)
                                + 5 * (-35 + 37 * b + 73 * b ** 2 + 57 * b ** 3)
                            )
                            + a ** 6
                            * a_conj ** 2
                            * (
                                a_conj ** 8 * (101 + 99 * b)
                                + a_conj ** 6 * (322 + 712 * b + 30 * b ** 2)
                                + (-1 + b) ** 3 * (103 + 154 * b + 93 * b ** 2)
                                - 12
                                * a_conj ** 4
                                * (13 - 63 * b - 54 * b ** 2 + 30 * b ** 3)
                                - 2
                                * a_conj ** 2
                                * (
                                    91
                                    + 180 * b
                                    - 162 * b ** 2
                                    - 124 * b ** 3
                                    + 15 * b ** 4
                                )
                            )
                            - a ** 4
                            * (
                                -3 * (-1 + b) ** 7
                                + 2 * a_conj ** 2 * (-1 + b) ** 5 * (-1 + 3 * b)
                                + 12
                                * a_conj ** 4
                                * (-1 + b) ** 3
                                * (1 + 9 * b + 3 * b ** 2)
                                - 5
                                * a_conj ** 8
                                * (-35 + 37 * b + 73 * b ** 2 + 57 * b ** 3)
                                + 2
                                * a_conj ** 6
                                * (
                                    91
                                    + 180 * b
                                    - 162 * b ** 2
                                    - 124 * b ** 3
                                    + 15 * b ** 4
                                )
                            )
                            + 2
                            * a ** 5
                            * (
                                a_conj * (-1 + b) ** 5 * (13 + 12 * b)
                                - 4
                                * a_conj ** 3
                                * (-1 + b) ** 3
                                * (-2 + b + 3 * b ** 2)
                                + 4 * a_conj ** 7 * (-11 + 76 * b + 85 * b ** 2)
                                + a_conj ** 9 * (29 + 167 * b + 114 * b ** 2)
                                - 6
                                * a_conj ** 5
                                * (
                                    17
                                    + 40 * b
                                    - 54 * b ** 2
                                    - 18 * b ** 3
                                    + 15 * b ** 4
                                )
                            )
                            + 2
                            * a ** 7
                            * a_conj ** 3
                            * (
                                9 * a_conj ** 8
                                + 4 * a_conj ** 6 * (31 + 3 * b)
                                + a_conj ** 4 * (156 + 324 * b - 180 * b ** 2)
                                + 4 * a_conj ** 2 * (-11 + 76 * b + 85 * b ** 2)
                                + 5 * (-23 - 6 * b + 8 * b ** 3 + 21 * b ** 4)
                            )
                        )
                        - 1
                        / (
                            b
                            * (
                                a ** 2 * a_conj ** 2
                                + 2 * a * a_conj * (b - x ** 2)
                                + (b + x ** 2) ** 2
                            )
                        )
                        * (
                            3
                            * a ** 2
                            * a_conj ** 2
                            * (a + a_conj) ** 4
                            * (
                                a ** 7 * a_conj ** 7
                                - a ** 6 * a_conj ** 6 * (-4 + 5 * b + x ** 2)
                                + a ** 5
                                * a_conj ** 5
                                * (6 - 35 * b ** 2 - 4 * x ** 2 - 2 * b * (-4 + x ** 2))
                                + a ** 4
                                * a_conj ** 4
                                * (
                                    4
                                    - 65 * b ** 3
                                    - 6 * x ** 2
                                    + b * (30 - 20 * x ** 2)
                                    + 5 * b ** 2 * (-4 + x ** 2)
                                )
                                + (-1 + b) ** 3
                                * b ** 2
                                * (5 * b ** 2 + 5 * x ** 2 + 3 * b * (1 + x ** 2))
                                + a ** 3
                                * a_conj ** 3
                                * (
                                    1
                                    - 45 * b ** 4
                                    - 4 * x ** 2
                                    + b ** 2 * (60 - 40 * x ** 2)
                                    + 20 * b ** 3 * (-4 + x ** 2)
                                    - 8 * b * (-2 + 3 * x ** 2)
                                )
                                + a ** 2
                                * a_conj ** 2
                                * (
                                    b ** 5
                                    - x ** 2
                                    + b ** 3 * (60 - 40 * x ** 2)
                                    + b ** 2 * (24 - 36 * x ** 2)
                                    + 25 * b ** 4 * (-4 + x ** 2)
                                    + b * (-1 + 4 * x ** 2)
                                )
                                + a
                                * a_conj
                                * (-1 + b)
                                * b
                                * (
                                    15 * b ** 4
                                    - 10 * x ** 2
                                    + b * (5 - 30 * x ** 2)
                                    - b ** 2 * (11 + 6 * x ** 2)
                                    + b ** 3 * (-41 + 14 * x ** 2)
                                )
                            )
                        )
                    )
                )
                + 1
                / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
                * (
                    24
                    * a ** 4
                    * a_conj ** 4
                    * (1 + a * a_conj + b)
                    * (
                        a ** 6 * a_conj ** 2 * (-5 + 3 * a_conj ** 2)
                        - 3 * (-1 + b) ** 4 * b
                        - 5 * a_conj ** 4 * (1 + 5 * b + b ** 2)
                        + 3 * a_conj ** 2 * (-1 + b) ** 2 * (1 + 8 * b + b ** 2)
                        + 2
                        * a ** 5
                        * (
                            3 * a_conj ** 5
                            - 5 * a_conj * (1 + b)
                            + a_conj ** 3 * (-4 + 6 * b)
                        )
                        + 4
                        * a ** 3
                        * a_conj
                        * (
                            -2
                            - 13 * b
                            + 7 * b ** 2
                            + 3 * b ** 3
                            + a_conj ** 4 * (-2 + 3 * b)
                            + 3 * a_conj ** 2 * (-2 + b + 2 * b ** 2)
                        )
                        + a ** 4
                        * (
                            3 * a_conj ** 6
                            + 3 * a_conj ** 4 * (-2 + 7 * b)
                            - 5 * (1 + 5 * b + b ** 2)
                            + 2 * a_conj ** 2 * (-11 + b + 9 * b ** 2)
                        )
                        - 2
                        * a
                        * (
                            5 * a_conj ** 5 * (1 + b)
                            + 3 * a_conj * (-1 + b) ** 2 * (-1 - 6 * b + b ** 2)
                            + a_conj ** 3 * (4 + 26 * b - 14 * b ** 2 - 6 * b ** 3)
                        )
                        + a ** 2
                        * (
                            -5 * a_conj ** 6
                            + 3 * (-1 + b) ** 2 * (1 + 8 * b + b ** 2)
                            + 2 * a_conj ** 4 * (-11 + b + 9 * b ** 2)
                            + 6 * a_conj ** 2 * (-1 - 12 * b + 9 * b ** 2 + b ** 3)
                        )
                    )
                    * np.log(
                        a * a_conj
                        - b
                        + (a * a_conj + b) ** 2
                        + np.sqrt((a * a_conj + b) ** 2)
                        * np.sqrt(1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2)
                    )
                )
                + 1
                / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
                * (
                    24
                    * a ** 4
                    * a_conj ** 4
                    * (1 + a * a_conj + b)
                    * (
                        a ** 6 * a_conj ** 2 * (-5 + 3 * a_conj ** 2)
                        - 3 * (-1 + b) ** 4 * b
                        - 5 * a_conj ** 4 * (1 + 5 * b + b ** 2)
                        + 3 * a_conj ** 2 * (-1 + b) ** 2 * (1 + 8 * b + b ** 2)
                        + 2
                        * a ** 5
                        * (
                            3 * a_conj ** 5
                            - 5 * a_conj * (1 + b)
                            + a_conj ** 3 * (-4 + 6 * b)
                        )
                        + 4
                        * a ** 3
                        * a_conj
                        * (
                            -2
                            - 13 * b
                            + 7 * b ** 2
                            + 3 * b ** 3
                            + a_conj ** 4 * (-2 + 3 * b)
                            + 3 * a_conj ** 2 * (-2 + b + 2 * b ** 2)
                        )
                        + a ** 4
                        * (
                            3 * a_conj ** 6
                            + 3 * a_conj ** 4 * (-2 + 7 * b)
                            - 5 * (1 + 5 * b + b ** 2)
                            + 2 * a_conj ** 2 * (-11 + b + 9 * b ** 2)
                        )
                        - 2
                        * a
                        * (
                            5 * a_conj ** 5 * (1 + b)
                            + 3 * a_conj * (-1 + b) ** 2 * (-1 - 6 * b + b ** 2)
                            + a_conj ** 3 * (4 + 26 * b - 14 * b ** 2 - 6 * b ** 3)
                        )
                        + a ** 2
                        * (
                            -5 * a_conj ** 6
                            + 3 * (-1 + b) ** 2 * (1 + 8 * b + b ** 2)
                            + 2 * a_conj ** 4 * (-11 + b + 9 * b ** 2)
                            + 6 * a_conj ** 2 * (-1 - 12 * b + 9 * b ** 2 + b ** 3)
                        )
                    )
                    * np.log(1 + x ** 2)
                )
                - 1
                / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
                * (
                    24
                    * a ** 4
                    * a_conj ** 4
                    * (1 + a * a_conj + b)
                    * (
                        a ** 6 * a_conj ** 2 * (-5 + 3 * a_conj ** 2)
                        - 3 * (-1 + b) ** 4 * b
                        - 5 * a_conj ** 4 * (1 + 5 * b + b ** 2)
                        + 3 * a_conj ** 2 * (-1 + b) ** 2 * (1 + 8 * b + b ** 2)
                        + 2
                        * a ** 5
                        * (
                            3 * a_conj ** 5
                            - 5 * a_conj * (1 + b)
                            + a_conj ** 3 * (-4 + 6 * b)
                        )
                        + 4
                        * a ** 3
                        * a_conj
                        * (
                            -2
                            - 13 * b
                            + 7 * b ** 2
                            + 3 * b ** 3
                            + a_conj ** 4 * (-2 + 3 * b)
                            + 3 * a_conj ** 2 * (-2 + b + 2 * b ** 2)
                        )
                        + a ** 4
                        * (
                            3 * a_conj ** 6
                            + 3 * a_conj ** 4 * (-2 + 7 * b)
                            - 5 * (1 + 5 * b + b ** 2)
                            + 2 * a_conj ** 2 * (-11 + b + 9 * b ** 2)
                        )
                        - 2
                        * a
                        * (
                            5 * a_conj ** 5 * (1 + b)
                            + 3 * a_conj * (-1 + b) ** 2 * (-1 - 6 * b + b ** 2)
                            + a_conj ** 3 * (4 + 26 * b - 14 * b ** 2 - 6 * b ** 3)
                        )
                        + a ** 2
                        * (
                            -5 * a_conj ** 6
                            + 3 * (-1 + b) ** 2 * (1 + 8 * b + b ** 2)
                            + 2 * a_conj ** 4 * (-11 + b + 9 * b ** 2)
                            + 6 * a_conj ** 2 * (-1 - 12 * b + 9 * b ** 2 + b ** 3)
                        )
                    )
                    * np.log(
                        a * a_conj
                        - b
                        + (a * a_conj + b) ** 2
                        - x ** 2
                        - a * a_conj * x ** 2
                        + b * x ** 2
                        + np.sqrt(1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2)
                        * np.sqrt(
                            -4 * a * a_conj * x ** 2 + (a * a_conj + b + x ** 2) ** 2
                        )
                    )
                )
            )
        )
        return 4 * np.pi * c * ret

    if abs(a_conj) < 0.00000001 and abs(b - 1) ** 2 >= 0.00000001:
        ret = (
            (-1 + b) ** 3
            + 3 * (-1 + b) * b
            - 3 * (-1 + b) ** 2 * b
            + 3 * (-1 + b) * b * (2 + b)
            - (-1 + b) ** 3 / (1 + x ** 2) ** 3
            + 3 * (-1 + b) ** 2 * b / (1 + x ** 2) ** 2
            - 3 * (-1 + b) * b * (2 + b) / (1 + x ** 2)
            - 3 * (-1 + b) * b ** 2 / (b + x ** 2)
            - 6 * b * (1 + b) * np.log(b)
            + 6 * b * (1 + b) * np.log(b + x ** 2)
            - 6 * b * (1 + b) * np.log(1 + x ** 2)
        ) / (6 * (-1 + b) ** 5)
        return 24 * np.pi * c * ret  # 12 as integral of 2cos^4 is 12pi
    if abs(a_conj) < 0.00000001 and abs(b - 1) ** 2 < 0.00000001:
        ret = x ** 6 * (10 + 5 * x ** 2 + x ** 4) / (60 * (1 + x ** 2) ** 5)
        return 24 * np.pi * c * ret


def xxxy_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return:
    """
    a_conj = np.conj(a)

    if abs(a_conj) >= 0.00000001:
        ret = (
            1
            / 12
            * (a ** 2 - a_conj ** 2)
            * (
                9 * (a ** 2 + a_conj ** 2) / (a ** 4 * a_conj ** 4)
                + (
                    3 * a ** 4 * a_conj ** 2
                    + a ** 2 * (-2 * a_conj ** 2 + 3 * a_conj ** 4 + 3 * (-1 + b) ** 2)
                    + 3 * a_conj ** 2 * (-1 + b) ** 2
                    + 2 * a ** 3 * a_conj * (-2 + 3 * b)
                    + 2 * a * a_conj ** 3 * (-2 + 3 * b)
                )
                / (a ** 4 * a_conj ** 4)
                + (
                    6 * a ** 3 * a_conj
                    + 6 * a * a_conj ** 3
                    + 9 * a_conj ** 2 * (-1 + b)
                    + 3 * a ** 2 * (-3 + a_conj ** 2 + 3 * b)
                )
                / (a ** 4 * a_conj ** 4)
                - 1
                / (
                    a ** 4
                    * a_conj ** 4
                    * b
                    * (a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b))
                    ** 4
                )
                * (
                    np.sqrt((a * a_conj + b) ** 2)
                    * (
                        3 * a ** 11 * a_conj ** 9 * b
                        + 3 * a_conj ** 2 * (-1 + b) ** 7 * b ** 3
                        + a ** 10 * a_conj ** 8 * b * (26 + 27 * b)
                        + a
                        * a_conj ** 3
                        * (-1 + b) ** 5
                        * b ** 2
                        * (9 + 14 * b + 27 * b ** 2)
                        + a ** 9
                        * a_conj ** 7
                        * (
                            -3
                            + (103 + a_conj ** 2 + 3 * a_conj ** 4) * b
                            + 161 * b ** 2
                            + 108 * b ** 3
                        )
                        + a ** 8
                        * a_conj ** 6
                        * (
                            -12
                            + 251 * b
                            + 409 * b ** 2
                            + 399 * b ** 3
                            + 252 * b ** 4
                            + a_conj ** 4 * b * (26 + 27 * b)
                            + a_conj ** 2 * (-6 + 8 * b + 7 * b ** 2)
                        )
                        + a ** 7
                        * a_conj ** 5
                        * (
                            -18
                            + 175 * b
                            + 601 * b ** 2
                            + 563 * b ** 3
                            + 469 * b ** 4
                            + 378 * b ** 5
                            + a_conj ** 2 * (-24 + 133 * b + 32 * b ** 2 + 21 * b ** 3)
                            + a_conj ** 4 * (-3 + 103 * b + 161 * b ** 2 + 108 * b ** 3)
                        )
                        + a ** 2
                        * b
                        * (
                            a_conj ** 2 * (-3 + b) * (-1 + b) ** 5 * b
                            + 3 * (-1 + b) ** 7 * b ** 2
                            + a_conj ** 4
                            * (-1 + b) ** 3
                            * (9 + 70 * b + 74 * b ** 2 + 65 * b ** 3 + 108 * b ** 4)
                        )
                        + a ** 6
                        * a_conj ** 4
                        * (
                            -12
                            - 112 * b
                            + 307 * b ** 2
                            + 407 * b ** 3
                            + 285 * b ** 4
                            + 175 * b ** 5
                            + 378 * b ** 6
                            + a_conj ** 2
                            * (-36 + 44 * b + 377 * b ** 2 + 40 * b ** 3 + 35 * b ** 4)
                            + a_conj ** 4
                            * (
                                -12
                                + 251 * b
                                + 409 * b ** 2
                                + 399 * b ** 3
                                + 252 * b ** 4
                            )
                        )
                        + a ** 5
                        * a_conj ** 3
                        * (
                            -3
                            - 125 * b
                            - 181 * b ** 2
                            + 96 * b ** 3
                            + 65 * b ** 4
                            + 85 * b ** 5
                            - 189 * b ** 6
                            + 252 * b ** 7
                            + a_conj ** 2
                            * (
                                -24
                                - 269 * b
                                + 368 * b ** 2
                                + 298 * b ** 3
                                + 35 * b ** 5
                            )
                            + a_conj ** 4
                            * (
                                -18
                                + 175 * b
                                + 601 * b ** 2
                                + 563 * b ** 3
                                + 469 * b ** 4
                                + 378 * b ** 5
                            )
                        )
                        + a ** 3
                        * a_conj
                        * (-1 + b)
                        * (
                            (-1 + b) ** 4 * b ** 2 * (9 + 14 * b + 27 * b ** 2)
                            + a_conj ** 2
                            * (-1 + b) ** 2
                            * b
                            * (-15 - 37 * b - 11 * b ** 2 + 7 * b ** 3)
                            + a_conj ** 4
                            * (
                                3
                                + 128 * b
                                + 309 * b ** 2
                                + 213 * b ** 3
                                + 148 * b ** 4
                                + 63 * b ** 5
                                + 252 * b ** 6
                            )
                        )
                        + a ** 4
                        * (
                            a_conj ** 2
                            * (-1 + b) ** 3
                            * b
                            * (9 + 70 * b + 74 * b ** 2 + 65 * b ** 3 + 108 * b ** 4)
                            + a_conj ** 4
                            * (
                                -6
                                - 172 * b
                                - 251 * b ** 2
                                + 396 * b ** 3
                                + 52 * b ** 4
                                - 40 * b ** 5
                                + 21 * b ** 6
                            )
                            + a_conj ** 6
                            * (
                                -12
                                - 112 * b
                                + 307 * b ** 2
                                + 407 * b ** 3
                                + 285 * b ** 4
                                + 175 * b ** 5
                                + 378 * b ** 6
                            )
                        )
                    )
                )
                + (
                    -3 * a ** 4 * a_conj ** 2
                    + a ** 2 * (2 * a_conj ** 2 - 3 * a_conj ** 4 - 3 * (-1 + b) ** 2)
                    + a ** 3 * a_conj * (4 - 6 * b)
                    + 2 * a * a_conj ** 3 * (2 - 3 * b)
                    - 3 * a_conj ** 2 * (-1 + b) ** 2
                )
                / (a ** 4 * a_conj ** 4 * (1 + x ** 2) ** 3)
                - 3
                * (
                    2 * a ** 3 * a_conj
                    + 2 * a * a_conj ** 3
                    + 3 * a_conj ** 2 * (-1 + b)
                    + a ** 2 * (-3 + a_conj ** 2 + 3 * b)
                )
                / (a ** 4 * a_conj ** 4 * (1 + x ** 2) ** 2)
                - 9 * (a ** 2 + a_conj ** 2) / (a ** 4 * a_conj ** 4 * (1 + x ** 2))
                + 1
                / (
                    a ** 4
                    * a_conj ** 4
                    * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** 4
                )
                * (
                    np.sqrt(
                        (a * a_conj + b) ** 2
                        - 2 * a * a_conj * x ** 2
                        + 2 * b * x ** 2
                        + x ** 4
                    )
                    * (
                        1
                        / (1 + x ** 2) ** 3
                        * (
                            (
                                a ** 2 * a_conj ** 2
                                + (-1 + b) ** 2
                                + 2 * a * a_conj * (1 + b)
                            )
                            ** 2
                            * (
                                3 * a ** 7 * a_conj ** 5
                                + 3 * a_conj ** 2 * (-1 + b) ** 5
                                + 5 * a ** 6 * a_conj ** 4 * (1 + 3 * b)
                                + 5 * a * a_conj ** 3 * (-1 + b) ** 3 * (1 + 3 * b)
                                + a ** 5
                                * (
                                    -2 * a_conj ** 5
                                    + 3 * a_conj ** 7
                                    + 30 * a_conj ** 3 * b ** 2
                                )
                                + a ** 2
                                * (-1 + b)
                                * (
                                    -2 * a_conj ** 2 * (-1 + b) ** 2
                                    + 3 * (-1 + b) ** 4
                                    + 30 * a_conj ** 4 * b ** 2
                                )
                                + a ** 4
                                * (
                                    30 * a_conj ** 2 * (-1 + b) * b ** 2
                                    - 6 * a_conj ** 4 * (1 + b)
                                    + 5 * a_conj ** 6 * (1 + 3 * b)
                                )
                                + a ** 3
                                * (
                                    30 * a_conj ** 5 * b ** 2
                                    + 5 * a_conj * (-1 + b) ** 3 * (1 + 3 * b)
                                    - 6 * a_conj ** 3 * (-1 + b ** 2)
                                )
                            )
                        )
                        + 1
                        / (1 + x ** 2) ** 2
                        * (
                            (
                                a ** 2 * a_conj ** 2
                                + (-1 + b) ** 2
                                + 2 * a * a_conj * (1 + b)
                            )
                            * (
                                9 * a ** 8 * a_conj ** 6
                                + 6 * a_conj ** 2 * (-1 + b) ** 6
                                + a * a_conj ** 3 * (-1 + b) ** 4 * (31 + 39 * b)
                                + a ** 7 * a_conj ** 5 * (29 + 3 * a_conj ** 2 + 51 * b)
                                + a ** 6
                                * (
                                    9 * a_conj ** 8
                                    + a_conj ** 6 * (13 + 15 * b)
                                    + 10 * a_conj ** 4 * (2 + 7 * b + 12 * b ** 2)
                                )
                                + a ** 2
                                * (
                                    6 * (-1 + b) ** 6
                                    + a_conj ** 2 * (-1 + b) ** 4 * (-1 + 3 * b)
                                    + 5
                                    * a_conj ** 4
                                    * (-1 + b) ** 2
                                    * (11 + 16 * b + 21 * b ** 2)
                                )
                                + a ** 5
                                * (
                                    a_conj ** 7 * (29 + 51 * b)
                                    + a_conj ** 5 * (-32 + 26 * b + 30 * b ** 2)
                                    + 30 * a_conj ** 3 * (1 + 5 * b ** 3)
                                )
                                + a ** 3
                                * (
                                    a_conj * (-1 + b) ** 4 * (31 + 39 * b)
                                    + a_conj ** 3
                                    * (-1 + b) ** 2
                                    * (-7 + 4 * b + 15 * b ** 2)
                                    + 30 * a_conj ** 5 * (1 + 5 * b ** 3)
                                )
                                + a ** 4
                                * (
                                    10 * a_conj ** 6 * (2 + 7 * b + 12 * b ** 2)
                                    + 5
                                    * a_conj ** 2
                                    * (-1 + b) ** 2
                                    * (11 + 16 * b + 21 * b ** 2)
                                    + 6 * a_conj ** 4 * (-8 - 9 * b + 5 * b ** 3)
                                )
                            )
                        )
                        + 1
                        / (1 + x ** 2)
                        * (
                            18 * a ** 9 * a_conj ** 7
                            + 3 * a_conj ** 2 * (-1 + b) ** 7
                            + 2 * a * a_conj ** 3 * (-1 + b) ** 5 * (13 + 12 * b)
                            + a ** 8 * a_conj ** 6 * (101 + 3 * a_conj ** 2 + 99 * b)
                            + 2
                            * a ** 7
                            * a_conj ** 5
                            * (
                                29
                                + 9 * a_conj ** 4
                                + 167 * b
                                + 114 * b ** 2
                                + a_conj ** 2 * (62 + 6 * b)
                            )
                            + a ** 2
                            * (
                                3 * (-1 + b) ** 7
                                - a_conj ** 2 * (-1 + b) ** 5 * (-1 + 3 * b)
                                + a_conj ** 4
                                * (-1 + b) ** 3
                                * (103 + 154 * b + 93 * b ** 2)
                            )
                            + 2
                            * a ** 3
                            * a_conj
                            * (-1 + b)
                            * (
                                (-1 + b) ** 4 * (13 + 12 * b)
                                - 2
                                * a_conj ** 2
                                * (-1 + b) ** 2
                                * (-2 + b + 3 * b ** 2)
                                + 5
                                * a_conj ** 4
                                * (23 + 29 * b + 29 * b ** 2 + 21 * b ** 3)
                            )
                            + a ** 6
                            * (
                                a_conj ** 8 * (101 + 99 * b)
                                + a_conj ** 6 * (161 + 356 * b + 15 * b ** 2)
                                + 5
                                * a_conj ** 4
                                * (-35 + 37 * b + 73 * b ** 2 + 57 * b ** 3)
                            )
                            + a ** 4
                            * (
                                a_conj ** 2
                                * (-1 + b) ** 3
                                * (103 + 154 * b + 93 * b ** 2)
                                + 5
                                * a_conj ** 6
                                * (-35 + 37 * b + 73 * b ** 2 + 57 * b ** 3)
                                + a_conj ** 4
                                * (
                                    -91
                                    - 180 * b
                                    + 162 * b ** 2
                                    + 124 * b ** 3
                                    - 15 * b ** 4
                                )
                            )
                            + 2
                            * a ** 5
                            * a_conj ** 3
                            * (
                                2 * a_conj ** 2 * (-11 + 76 * b + 85 * b ** 2)
                                + a_conj ** 4 * (29 + 167 * b + 114 * b ** 2)
                                + 5 * (-23 - 6 * b + 8 * b ** 3 + 21 * b ** 4)
                            )
                        )
                        - 1
                        / (
                            b
                            * (
                                a ** 2 * a_conj ** 2
                                + 2 * a * a_conj * (b - x ** 2)
                                + (b + x ** 2) ** 2
                            )
                        )
                        * (
                            3
                            * a ** 2
                            * a_conj ** 2
                            * (a + a_conj) ** 2
                            * (
                                a ** 7 * a_conj ** 7
                                - a ** 6 * a_conj ** 6 * (-4 + 5 * b + x ** 2)
                                + a ** 5
                                * a_conj ** 5
                                * (6 - 35 * b ** 2 - 4 * x ** 2 - 2 * b * (-4 + x ** 2))
                                + a ** 4
                                * a_conj ** 4
                                * (
                                    4
                                    - 65 * b ** 3
                                    - 6 * x ** 2
                                    + b * (30 - 20 * x ** 2)
                                    + 5 * b ** 2 * (-4 + x ** 2)
                                )
                                + (-1 + b) ** 3
                                * b ** 2
                                * (5 * b ** 2 + 5 * x ** 2 + 3 * b * (1 + x ** 2))
                                + a ** 3
                                * a_conj ** 3
                                * (
                                    1
                                    - 45 * b ** 4
                                    - 4 * x ** 2
                                    + b ** 2 * (60 - 40 * x ** 2)
                                    + 20 * b ** 3 * (-4 + x ** 2)
                                    - 8 * b * (-2 + 3 * x ** 2)
                                )
                                + a ** 2
                                * a_conj ** 2
                                * (
                                    b ** 5
                                    - x ** 2
                                    + b ** 3 * (60 - 40 * x ** 2)
                                    + b ** 2 * (24 - 36 * x ** 2)
                                    + 25 * b ** 4 * (-4 + x ** 2)
                                    + b * (-1 + 4 * x ** 2)
                                )
                                + a
                                * a_conj
                                * (-1 + b)
                                * b
                                * (
                                    15 * b ** 4
                                    - 10 * x ** 2
                                    + b * (5 - 30 * x ** 2)
                                    - b ** 2 * (11 + 6 * x ** 2)
                                    + b ** 3 * (-41 + 14 * x ** 2)
                                )
                            )
                        )
                    )
                )
                + 1
                / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
                * (
                    12
                    * (1 + a * a_conj + b)
                    * (
                        a ** 4 * a_conj ** 2 * (-10 + 3 * a_conj ** 2)
                        - 10 * a_conj ** 2 * (1 + 5 * b + b ** 2)
                        + 3 * (-1 + b) ** 2 * (1 + 8 * b + b ** 2)
                        - 4
                        * a
                        * a_conj
                        * (
                            2
                            + 13 * b
                            - 7 * b ** 2
                            - 3 * b ** 3
                            + 5 * a_conj ** 2 * (1 + b)
                        )
                        + 4
                        * a ** 3
                        * (-5 * a_conj * (1 + b) + a_conj ** 3 * (-2 + 3 * b))
                        - 2
                        * a ** 2
                        * (
                            5 * a_conj ** 4
                            + 5 * (1 + 5 * b + b ** 2)
                            - a_conj ** 2 * (-11 + b + 9 * b ** 2)
                        )
                    )
                    * np.log(
                        a * a_conj
                        - b
                        + (a * a_conj + b) ** 2
                        + np.sqrt((a * a_conj + b) ** 2)
                        * np.sqrt(1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2)
                    )
                )
                + 1
                / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
                * (
                    12
                    * (1 + a * a_conj + b)
                    * (
                        a ** 4 * a_conj ** 2 * (-10 + 3 * a_conj ** 2)
                        - 10 * a_conj ** 2 * (1 + 5 * b + b ** 2)
                        + 3 * (-1 + b) ** 2 * (1 + 8 * b + b ** 2)
                        - 4
                        * a
                        * a_conj
                        * (
                            2
                            + 13 * b
                            - 7 * b ** 2
                            - 3 * b ** 3
                            + 5 * a_conj ** 2 * (1 + b)
                        )
                        + 4
                        * a ** 3
                        * (-5 * a_conj * (1 + b) + a_conj ** 3 * (-2 + 3 * b))
                        - 2
                        * a ** 2
                        * (
                            5 * a_conj ** 4
                            + 5 * (1 + 5 * b + b ** 2)
                            - a_conj ** 2 * (-11 + b + 9 * b ** 2)
                        )
                    )
                    * np.log(1 + x ** 2)
                )
                - 1
                / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
                * (
                    12
                    * (1 + a * a_conj + b)
                    * (
                        a ** 4 * a_conj ** 2 * (-10 + 3 * a_conj ** 2)
                        - 10 * a_conj ** 2 * (1 + 5 * b + b ** 2)
                        + 3 * (-1 + b) ** 2 * (1 + 8 * b + b ** 2)
                        - 4
                        * a
                        * a_conj
                        * (
                            2
                            + 13 * b
                            - 7 * b ** 2
                            - 3 * b ** 3
                            + 5 * a_conj ** 2 * (1 + b)
                        )
                        + 4
                        * a ** 3
                        * (-5 * a_conj * (1 + b) + a_conj ** 3 * (-2 + 3 * b))
                        - 2
                        * a ** 2
                        * (
                            5 * a_conj ** 4
                            + 5 * (1 + 5 * b + b ** 2)
                            - a_conj ** 2 * (-11 + b + 9 * b ** 2)
                        )
                    )
                    * np.log(
                        a * a_conj
                        - b
                        + (a * a_conj + b) ** 2
                        - x ** 2
                        - a * a_conj * x ** 2
                        + b * x ** 2
                        + np.sqrt(1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2)
                        * np.sqrt(
                            a ** 2 * a_conj ** 2
                            + 2 * a * a_conj * (b - x ** 2)
                            + (b + x ** 2) ** 2
                        )
                    )
                )
            )
        )
        return 4 / 1j * np.pi * c * ret
    if abs(a_conj) < 0.00000001:
        return 0


def xxzz_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return:
    """
    a_conj = np.conj(a)

    if abs(a_conj) >= 0.00000001:
        ret = (
            1
            / 6
            * (
                1 / a ** 2
                + 1 / a_conj ** 2
                - 1
                / (
                    a ** 2
                    * a_conj ** 2
                    * b
                    * (a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b))
                    ** 4
                )
                * (
                    np.sqrt((a * a_conj + b) ** 2)
                    * (
                        a ** 9 * a_conj ** 7 * (-3 + b)
                        - a ** 8 * a_conj ** 6 * (6 + 6 * a_conj ** 2 + b - 7 * b ** 2)
                        + a_conj ** 2 * (-1 + b) ** 5 * b * (3 + b ** 2)
                        + a
                        * a_conj ** 3
                        * (-1 + b) ** 3
                        * (3 + 54 * b + 38 * b ** 2 + 10 * b ** 3 + 7 * b ** 4)
                        + a ** 7
                        * a_conj ** 5
                        * (
                            3
                            + a_conj ** 4 * (-3 + b)
                            - 239 * b
                            + 35 * b ** 2
                            + 21 * b ** 3
                            + 4 * a_conj ** 2 * (-3 + 4 * b)
                        )
                        + a ** 6
                        * a_conj ** 4
                        * (
                            12
                            - 73 * b
                            - 691 * b ** 2
                            + 85 * b ** 3
                            + 35 * b ** 4
                            + a_conj ** 4 * (-6 - b + 7 * b ** 2)
                            + 2 * a_conj ** 2 * (3 - 220 * b + 85 * b ** 2)
                        )
                        + a ** 5
                        * a_conj ** 3
                        * (
                            3
                            + 523 * b
                            - 766 * b ** 2
                            - 686 * b ** 3
                            + 75 * b ** 4
                            + 35 * b ** 5
                            + a_conj ** 4 * (3 - 239 * b + 35 * b ** 2 + 21 * b ** 3)
                            + 8
                            * a_conj ** 2
                            * (3 - 26 * b - 147 * b ** 2 + 50 * b ** 3)
                        )
                        + a ** 4
                        * a_conj ** 2
                        * (
                            -6
                            + 317 * b
                            + 565 * b ** 2
                            - 702 * b ** 3
                            - 212 * b ** 4
                            + 17 * b ** 5
                            + 21 * b ** 6
                            + a_conj ** 4
                            * (12 - 73 * b - 691 * b ** 2 + 85 * b ** 3 + 35 * b ** 4)
                            + 2
                            * a_conj ** 2
                            * (3 + 424 * b - 730 * b ** 2 - 504 * b ** 3 + 215 * b ** 4)
                        )
                        + a ** 2
                        * (-1 + b)
                        * (
                            (-1 + b) ** 4 * b * (3 + b ** 2)
                            + 2
                            * a_conj ** 2
                            * (-1 + b) ** 2
                            * (3 + 69 * b + 105 * b ** 2 + 23 * b ** 3)
                            + a_conj ** 4
                            * (
                                6
                                - 311 * b
                                - 876 * b ** 2
                                - 174 * b ** 3
                                + 38 * b ** 4
                                + 21 * b ** 5
                            )
                        )
                        + a ** 3
                        * (
                            a_conj
                            * (-1 + b) ** 3
                            * (3 + 54 * b + 38 * b ** 2 + 10 * b ** 3 + 7 * b ** 4)
                            + a_conj ** 5
                            * (
                                3
                                + 523 * b
                                - 766 * b ** 2
                                - 686 * b ** 3
                                + 75 * b ** 4
                                + 35 * b ** 5
                            )
                            + 4
                            * a_conj ** 3
                            * (
                                -3
                                + 120 * b
                                + 274 * b ** 2
                                - 400 * b ** 3
                                - 47 * b ** 4
                                + 56 * b ** 5
                            )
                        )
                    )
                )
                - 4
                * (a ** 2 + a_conj ** 2)
                / (a ** 2 * a_conj ** 2 * (1 + x ** 2) ** 3)
                + 6
                * (a ** 2 + a_conj ** 2)
                / (a ** 2 * a_conj ** 2 * (1 + x ** 2) ** 2)
                - 3 * (a ** 2 + a_conj ** 2) / (a ** 2 * a_conj ** 2 * (1 + x ** 2))
                + 1
                / (
                    a ** 2
                    * a_conj ** 2
                    * (a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b))
                    ** 4
                )
                * (
                    np.sqrt(
                        (a * a_conj + b) ** 2
                        - 2 * a * a_conj * x ** 2
                        + 2 * b * x ** 2
                        + x ** 4
                    )
                    * (
                        4
                        * (
                            a ** 2 * a_conj ** 2
                            + (-1 + b) ** 2
                            + 2 * a * a_conj * (1 + b)
                        )
                        ** 2
                        * (
                            a ** 5 * a_conj ** 3
                            + a_conj ** 2 * (-1 + b) ** 3
                            + 3 * a ** 4 * a_conj ** 2 * (1 + b)
                            + 3 * a * a_conj ** 3 * (-1 + b ** 2)
                            + a ** 3
                            * a_conj
                            * (-3 + 4 * a_conj ** 2 + a_conj ** 4 + 3 * b ** 2)
                            + a ** 2
                            * (
                                4 * a_conj ** 2 * (-1 + b)
                                + (-1 + b) ** 3
                                + 3 * a_conj ** 4 * (1 + b)
                            )
                        )
                        / (1 + x ** 2) ** 3
                        - 1
                        / (1 + x ** 2) ** 2
                        * (
                            2
                            * (
                                a ** 2 * a_conj ** 2
                                + (-1 + b) ** 2
                                + 2 * a * a_conj * (1 + b)
                            )
                            * (
                                3 * a ** 7 * a_conj ** 5
                                + a_conj ** 2 * (-1 + b) ** 4 * (-1 + 3 * b)
                                + a ** 6 * a_conj ** 4 * (13 + 15 * b)
                                + a
                                * a_conj ** 3
                                * (-1 + b) ** 2
                                * (-7 + 4 * b + 15 * b ** 2)
                                + a ** 5
                                * a_conj ** 3
                                * (
                                    -32
                                    + 24 * a_conj ** 2
                                    + 3 * a_conj ** 4
                                    + 26 * b
                                    + 30 * b ** 2
                                )
                                + a ** 3
                                * (
                                    24 * a_conj ** 3 * (-2 - 3 * b + 3 * b ** 2)
                                    + a_conj
                                    * (-1 + b) ** 2
                                    * (-7 + 4 * b + 15 * b ** 2)
                                    + a_conj ** 5 * (-32 + 26 * b + 30 * b ** 2)
                                )
                                + a ** 4
                                * (
                                    8 * a_conj ** 4 * (-4 + 9 * b)
                                    + a_conj ** 6 * (13 + 15 * b)
                                    + 6 * a_conj ** 2 * (-8 - 9 * b + 5 * b ** 3)
                                )
                                + a ** 2
                                * (
                                    (-1 + b) ** 4 * (-1 + 3 * b)
                                    + 8 * a_conj ** 2 * (-1 + b) ** 2 * (1 + 3 * b)
                                    + 6 * a_conj ** 4 * (-8 - 9 * b + 5 * b ** 3)
                                )
                            )
                        )
                        + 1
                        / (1 + x ** 2)
                        * (
                            3 * a ** 9 * a_conj ** 7
                            + 3 * a ** 8 * a_conj ** 6 * (5 + 7 * b)
                            + a_conj ** 2 * (-1 + b) ** 5 * (1 + 3 * b ** 2)
                            + a ** 7
                            * a_conj ** 5
                            * (
                                -203
                                + 60 * a_conj ** 2
                                + 3 * a_conj ** 4
                                + 60 * b
                                + 63 * b ** 2
                            )
                            + a
                            * a_conj ** 3
                            * (-1 + b) ** 3
                            * (5 - 13 * b + 3 * b ** 2 + 21 * b ** 3)
                            + a ** 6
                            * a_conj ** 4
                            * (
                                -295
                                - 661 * b
                                + 75 * b ** 2
                                + 105 * b ** 3
                                + 3 * a_conj ** 4 * (5 + 7 * b)
                                + 12 * a_conj ** 2 * (-21 + 25 * b)
                            )
                            + a ** 5
                            * a_conj ** 3
                            * (
                                61
                                - 632 * b
                                - 734 * b ** 2
                                + 105 * b ** 4
                                + a_conj ** 4 * (-203 + 60 * b + 63 * b ** 2)
                                + 8 * a_conj ** 2 * (-49 - 102 * b + 75 * b ** 2)
                            )
                            + a ** 4
                            * a_conj ** 2
                            * (
                                137
                                + 411 * b
                                - 270 * b ** 2
                                - 266 * b ** 3
                                - 75 * b ** 4
                                + 63 * b ** 5
                                + 8
                                * a_conj ** 2
                                * (23 - 129 * b - 117 * b ** 2 + 75 * b ** 3)
                                + a_conj ** 4
                                * (-295 - 661 * b + 75 * b ** 2 + 105 * b ** 3)
                            )
                            + a ** 2
                            * (-1 + b)
                            * (
                                (-1 + b) ** 4 * (1 + 3 * b ** 2)
                                + 4
                                * a_conj ** 2
                                * (-1 + b) ** 2
                                * (7 + 30 * b + 15 * b ** 2)
                                + a_conj ** 4
                                * (
                                    -137
                                    - 548 * b
                                    - 278 * b ** 2
                                    - 12 * b ** 3
                                    + 63 * b ** 4
                                )
                            )
                            + a ** 3
                            * (
                                a_conj
                                * (-1 + b) ** 3
                                * (5 - 13 * b + 3 * b ** 2 + 21 * b ** 3)
                                + 4
                                * a_conj ** 3
                                * (
                                    59
                                    + 172 * b
                                    - 198 * b ** 2
                                    - 108 * b ** 3
                                    + 75 * b ** 4
                                )
                                + a_conj ** 5
                                * (61 - 632 * b - 734 * b ** 2 + 105 * b ** 4)
                            )
                        )
                        - 1
                        / (
                            b
                            * (
                                a ** 2 * a_conj ** 2
                                + 2 * a * a_conj * (b - x ** 2)
                                + (b + x ** 2) ** 2
                            )
                        )
                        * (
                            3
                            * a
                            * a_conj
                            * (a + a_conj) ** 2
                            * (
                                a ** 8 * a_conj ** 8
                                + a ** 7 * a_conj ** 7 * (2 + 4 * b - x ** 2)
                                - (-1 + b) ** 3
                                * b
                                * (1 + b)
                                * (
                                    b
                                    + 3 * b ** 3
                                    + 3 * x ** 2
                                    + 12 * b * x ** 2
                                    + b ** 2 * (12 + x ** 2)
                                )
                                + a ** 5
                                * a_conj ** 5
                                * (
                                    -4
                                    - 28 * b ** 3
                                    + x ** 2
                                    + b * (2 - 20 * x ** 2)
                                    - 7 * b ** 2 * (-22 + 3 * x ** 2)
                                )
                                - a ** 4
                                * a_conj ** 4
                                * (
                                    1
                                    + 70 * b ** 4
                                    - 4 * x ** 2
                                    + b * (84 - 41 * x ** 2)
                                    + 35 * b ** 2 * (-3 + 2 * x ** 2)
                                    + 5 * b ** 3 * (-54 + 7 * x ** 2)
                                )
                                - a ** 6
                                * a_conj ** 6
                                * (1 + 2 * x ** 2 + b * (-38 + 7 * x ** 2))
                                - a
                                * a_conj
                                * (-1 + b)
                                * (
                                    20 * b ** 6
                                    - x ** 2
                                    + b ** 2 * (24 - 122 * x ** 2)
                                    + b * (2 - 29 * x ** 2)
                                    - 14 * b ** 3 * (4 + 3 * x ** 2)
                                    + b ** 5 * (22 + 7 * x ** 2)
                                    + b ** 4 * (-140 + 59 * x ** 2)
                                )
                                + a ** 3
                                * a_conj ** 3
                                * (
                                    2
                                    - 84 * b ** 5
                                    + x ** 2
                                    + 40 * b * (-1 + 2 * x ** 2)
                                    - 5 * b ** 4 * (-46 + 7 * x ** 2)
                                    - 4 * b ** 3 * (-79 + 30 * x ** 2)
                                    + 2 * b ** 2 * (-116 + 69 * x ** 2)
                                )
                                + a ** 2
                                * a_conj ** 2
                                * (
                                    1
                                    - 56 * b ** 6
                                    - 2 * x ** 2
                                    + b ** 4 * (353 - 110 * x ** 2)
                                    + b ** 5 * (82 - 21 * x ** 2)
                                    + b * (14 - 5 * x ** 2)
                                    + 2 * b ** 2 * (-49 + 76 * x ** 2)
                                    + 2 * b ** 3 * (-116 + 89 * x ** 2)
                                )
                            )
                        )
                    )
                )
                + 1
                / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
                * (
                    12
                    * (1 + a * a_conj + b)
                    * (
                        a ** 6 * a_conj ** 4 * (-9 + a_conj ** 2)
                        + (-1 + b) ** 4 * (1 + 6 * b + b ** 2)
                        - 3 * a_conj ** 2 * (-1 + b) ** 2 * (3 + 14 * b + 3 * b ** 2)
                        + 2
                        * a
                        * a_conj
                        * (-1 + b) ** 2
                        * (-8 - 45 * b + 2 * b ** 2 + 3 * b ** 3)
                        - 4 * a * a_conj ** 3 * (-1 - 29 * b + 11 * b ** 2 + 9 * b ** 3)
                        + 2
                        * a ** 5
                        * a_conj ** 3
                        * (2 - 18 * b + a_conj ** 2 * (-8 + 3 * b))
                        + a ** 4
                        * a_conj ** 2
                        * (
                            26
                            - 9 * a_conj ** 4
                            - 16 * b
                            - 54 * b ** 2
                            + a_conj ** 2 * (7 - 62 * b + 15 * b ** 2)
                        )
                        - 4
                        * a ** 3
                        * a_conj
                        * (
                            -1
                            - 29 * b
                            + 11 * b ** 2
                            + 9 * b ** 3
                            + a_conj ** 4 * (-1 + 9 * b)
                            + a_conj ** 2 * (-12 + 11 * b + 22 * b ** 2 - 5 * b ** 3)
                        )
                        - a ** 2
                        * (
                            3 * (-1 + b) ** 2 * (3 + 14 * b + 3 * b ** 2)
                            + 2 * a_conj ** 4 * (-13 + 8 * b + 27 * b ** 2)
                            + a_conj ** 2
                            * (-7 - 204 * b + 126 * b ** 2 + 52 * b ** 3 - 15 * b ** 4)
                        )
                    )
                    * np.arctanh(
                        (
                            a ** 2 * a_conj ** 2
                            + (-1 + b) * b
                            + a * (a_conj + 2 * a_conj * b)
                        )
                        / (
                            np.sqrt((a * a_conj + b) ** 2)
                            * np.sqrt(
                                a ** 2 * a_conj ** 2
                                + (-1 + b) ** 2
                                + 2 * a * a_conj * (1 + b)
                            )
                        )
                    )
                )
                - 1
                / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
                * (
                    12
                    * (1 + a * a_conj + b)
                    * (
                        a ** 6 * a_conj ** 4 * (-9 + a_conj ** 2)
                        + (-1 + b) ** 4 * (1 + 6 * b + b ** 2)
                        - 3 * a_conj ** 2 * (-1 + b) ** 2 * (3 + 14 * b + 3 * b ** 2)
                        + 2
                        * a
                        * a_conj
                        * (-1 + b) ** 2
                        * (-8 - 45 * b + 2 * b ** 2 + 3 * b ** 3)
                        - 4 * a * a_conj ** 3 * (-1 - 29 * b + 11 * b ** 2 + 9 * b ** 3)
                        + 2
                        * a ** 5
                        * a_conj ** 3
                        * (2 - 18 * b + a_conj ** 2 * (-8 + 3 * b))
                        + a ** 4
                        * a_conj ** 2
                        * (
                            26
                            - 9 * a_conj ** 4
                            - 16 * b
                            - 54 * b ** 2
                            + a_conj ** 2 * (7 - 62 * b + 15 * b ** 2)
                        )
                        - 4
                        * a ** 3
                        * a_conj
                        * (
                            -1
                            - 29 * b
                            + 11 * b ** 2
                            + 9 * b ** 3
                            + a_conj ** 4 * (-1 + 9 * b)
                            + a_conj ** 2 * (-12 + 11 * b + 22 * b ** 2 - 5 * b ** 3)
                        )
                        - a ** 2
                        * (
                            3 * (-1 + b) ** 2 * (3 + 14 * b + 3 * b ** 2)
                            + 2 * a_conj ** 4 * (-13 + 8 * b + 27 * b ** 2)
                            + a_conj ** 2
                            * (-7 - 204 * b + 126 * b ** 2 + 52 * b ** 3 - 15 * b ** 4)
                        )
                    )
                    * np.arctanh(
                        (
                            a ** 2 * a_conj ** 2
                            + a * a_conj * (1 + 2 * b - x ** 2)
                            + (-1 + b) * (b + x ** 2)
                        )
                        / (
                            np.sqrt(
                                a ** 2 * a_conj ** 2
                                + (-1 + b) ** 2
                                + 2 * a * a_conj * (1 + b)
                            )
                            * np.sqrt(
                                a ** 2 * a_conj ** 2
                                + 2 * a * a_conj * (b - x ** 2)
                                + (b + x ** 2) ** 2
                            )
                        )
                    )
                )
            )
        )
        return 2 * np.pi * c * ret
    if abs(a_conj) < 0.00000001 and (b - 1) ** 2 >= 0.00000001:
        ret = (
            1
            / (6 * (-1 + b) ** 5)
            * (
                -4 * (-1 + b) ** 3
                + 12 * (-1 + b) ** 2 * b
                - 3 * (-1 + b) * (1 + b) ** 2
                - 3 * (-1 + b) * (1 + b) * (1 + 5 * b)
                + 4 * (-1 + b) ** 3 / (1 + x ** 2) ** 3
                - 12 * (-1 + b) ** 2 * b / (1 + x ** 2) ** 2
                + 3 * (-1 + b) * (1 + b) * (1 + 5 * b) / (1 + x ** 2)
                + 3 * (-1 + b) * b * (1 + b) ** 2 / (b + x ** 2)
                + 3 * (1 + 7 * b + 7 * b ** 2 + b ** 3) * np.log(b)
                + 3 * (1 + 7 * b + 7 * b ** 2 + b ** 3) * np.log(1 + x ** 2)
                - 3 * (1 + 7 * b + 7 * b ** 2 + b ** 3) * np.log(b + x ** 2)
            )
        )
        return 8 * np.pi * c * ret
    if abs(a_conj) < 0.00000001 and (b - 1) ** 2 < 0.00000001:
        ret = (
            x ** 4 * (15 - 5 * x ** 2 + 5 * x ** 4 + x ** 6) / (60 * (1 + x ** 2) ** 5)
        )
        return 8 * np.pi * c * ret


def xzzz_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return:
    """
    a_conj = np.conj(a)

    if abs(a_conj) >= 0.00000001:
        ret = (
            1
            / (
                6
                * a
                * a_conj
                * b
                * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
            )
            * (
                (a + a_conj)
                * (
                    -54
                    * (
                        a ** 2 * a_conj ** 2
                        + 2 * a * a_conj * (1 - 5 * b)
                        + (-1 + b) ** 2
                    )
                    * np.sqrt((a * a_conj + b) ** 2)
                    * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (5 / 2)
                    - 8
                    * (
                        3 * a ** 2 * a_conj ** 2
                        + a * a_conj * (6 - 22 * b)
                        + 3 * (-1 + b) ** 2
                    )
                    * np.sqrt((a * a_conj + b) ** 2)
                    * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (5 / 2)
                    + 3
                    * (a * a_conj - b)
                    * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
                    / np.sqrt((a * a_conj + b) ** 2)
                    - 3
                    * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (7 / 2)
                    * (
                        a ** 2 * a_conj ** 2
                        + (-1 + b) * b
                        + a * (a_conj - 6 * a_conj * b)
                    )
                    / np.sqrt((a * a_conj + b) ** 2)
                    - 8
                    * np.sqrt((a * a_conj + b) ** 2)
                    * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (3 / 2)
                    * (
                        3 * a ** 3 * a_conj ** 3
                        + a ** 2 * a_conj ** 2 * (9 - 67 * b)
                        - 3 * (-1 + b) ** 3
                        + a * a_conj * (9 - 76 * b + 67 * b ** 2)
                    )
                    - 8
                    * np.sqrt((a * a_conj + b) ** 2)
                    * np.sqrt(1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2)
                    * (
                        3 * a ** 4 * a_conj ** 4
                        + 3 * (-1 + b) ** 4
                        - 2 * a ** 3 * a_conj ** 3 * (-6 + 89 * b)
                        - 2 * a * a_conj * (-1 + b) ** 2 * (-6 + 89 * b)
                        + 2 * a ** 2 * a_conj ** 2 * (9 - 184 * b + 239 * b ** 2)
                    )
                    - 3
                    * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
                    * (a * a_conj - b - x ** 2)
                    / np.sqrt(
                        (a * a_conj + b) ** 2 - 2 * (a * a_conj - b) * x ** 2 + x ** 4
                    )
                    + 8
                    * (
                        3 * a ** 2 * a_conj ** 2
                        + a * a_conj * (6 - 22 * b)
                        + 3 * (-1 + b) ** 2
                    )
                    * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (5 / 2)
                    * np.sqrt(
                        (a * a_conj + b) ** 2 - 2 * (a * a_conj - b) * x ** 2 + x ** 4
                    )
                    / (1 + x ** 2) ** 3
                    + 8
                    * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (3 / 2)
                    * (
                        3 * a ** 3 * a_conj ** 3
                        + a ** 2 * a_conj ** 2 * (9 - 67 * b)
                        - 3 * (-1 + b) ** 3
                        + a * a_conj * (9 - 76 * b + 67 * b ** 2)
                    )
                    * np.sqrt(
                        (a * a_conj + b) ** 2 - 2 * (a * a_conj - b) * x ** 2 + x ** 4
                    )
                    / (1 + x ** 2) ** 2
                    + 54
                    * (
                        a ** 2 * a_conj ** 2
                        + 2 * a * a_conj * (1 - 5 * b)
                        + (-1 + b) ** 2
                    )
                    * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (5 / 2)
                    * np.sqrt(
                        (a * a_conj + b) ** 2 - 2 * (a * a_conj - b) * x ** 2 + x ** 4
                    )
                    / (1 + x ** 2)
                    + 8
                    * np.sqrt(1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2)
                    * (
                        3 * a ** 4 * a_conj ** 4
                        + 3 * (-1 + b) ** 4
                        - 2 * a ** 3 * a_conj ** 3 * (-6 + 89 * b)
                        - 2 * a * a_conj * (-1 + b) ** 2 * (-6 + 89 * b)
                        + 2 * a ** 2 * a_conj ** 2 * (9 - 184 * b + 239 * b ** 2)
                    )
                    * np.sqrt(
                        (a * a_conj + b) ** 2 - 2 * (a * a_conj - b) * x ** 2 + x ** 4
                    )
                    / (1 + x ** 2)
                    + 21
                    * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (7 / 2)
                    * (
                        a ** 2 * a_conj ** 2
                        + b ** 2
                        - x ** 2
                        + b * (-1 + x ** 2)
                        - a * a_conj * (-1 + 6 * b + x ** 2)
                    )
                    / np.sqrt(
                        (a * a_conj + b) ** 2 - 2 * (a * a_conj - b) * x ** 2 + x ** 4
                    )
                    - 24
                    * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (7 / 2)
                    * (
                        a ** 2 * a_conj ** 2
                        + b ** 2
                        - x ** 2
                        + b * (-1 + x ** 2)
                        - a * a_conj * (-1 + 6 * b + x ** 2)
                    )
                    / (
                        (1 + x ** 2) ** 3
                        * np.sqrt(
                            (a * a_conj + b) ** 2
                            - 2 * (a * a_conj - b) * x ** 2
                            + x ** 4
                        )
                    )
                    + 60
                    * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (7 / 2)
                    * (
                        a ** 2 * a_conj ** 2
                        + b ** 2
                        - x ** 2
                        + b * (-1 + x ** 2)
                        - a * a_conj * (-1 + 6 * b + x ** 2)
                    )
                    / (
                        (1 + x ** 2) ** 2
                        * np.sqrt(
                            (a * a_conj + b) ** 2
                            - 2 * (a * a_conj - b) * x ** 2
                            + x ** 4
                        )
                    )
                    - 54
                    * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (7 / 2)
                    * (
                        a ** 2 * a_conj ** 2
                        + b ** 2
                        - x ** 2
                        + b * (-1 + x ** 2)
                        - a * a_conj * (-1 + 6 * b + x ** 2)
                    )
                    / (
                        (1 + x ** 2)
                        * np.sqrt(
                            (a * a_conj + b) ** 2
                            - 2 * (a * a_conj - b) * x ** 2
                            + x ** 4
                        )
                    )
                    + 648
                    * a
                    * a_conj
                    * (1 + a * a_conj - b)
                    * b
                    * (a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b))
                    ** 2
                    * np.arctanh(
                        (
                            a ** 2 * a_conj ** 2
                            + (-1 + b) * b
                            + a * (a_conj + 2 * a_conj * b)
                        )
                        / (
                            np.sqrt((a * a_conj + b) ** 2)
                            * np.sqrt(
                                a ** 2 * a_conj ** 2
                                + (-1 + b) ** 2
                                + 2 * a * a_conj * (1 + b)
                            )
                        )
                    )
                    - 84
                    * a
                    * a_conj
                    * b
                    * (a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b))
                    ** 3
                    * np.arctanh(
                        (
                            a ** 2 * a_conj ** 2
                            + (-1 + b) * b
                            + a * (a_conj + 2 * a_conj * b)
                        )
                        / (
                            np.sqrt((a * a_conj + b) ** 2)
                            * np.sqrt(
                                a ** 2 * a_conj ** 2
                                + (-1 + b) ** 2
                                + 2 * a * a_conj * (1 + b)
                            )
                        )
                    )
                    + 960
                    * a
                    * a_conj
                    * b
                    * (
                        a ** 3 * a_conj ** 3
                        + 3 * a ** 2 * a_conj ** 2 * (1 - 2 * b)
                        - (-1 + b) ** 3
                        + 3 * a * a_conj * (1 - 3 * b + 2 * b ** 2)
                    )
                    * np.arctanh(
                        (
                            a ** 2 * a_conj ** 2
                            + (-1 + b) * b
                            + a * (a_conj + 2 * a_conj * b)
                        )
                        / (
                            np.sqrt((a * a_conj + b) ** 2)
                            * np.sqrt(
                                a ** 2 * a_conj ** 2
                                + (-1 + b) ** 2
                                + 2 * a * a_conj * (1 + b)
                            )
                        )
                    )
                    + 60
                    * (a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b))
                    * (
                        (
                            a ** 2 * a_conj ** 2
                            + a * a_conj * (2 - 8 * b)
                            + (-1 + b) ** 2
                        )
                        * np.sqrt((a * a_conj + b) ** 2)
                        * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2)
                        ** (3 / 2)
                        + np.sqrt((a * a_conj + b) ** 2)
                        * np.sqrt(1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2)
                        * (
                            a ** 3 * a_conj ** 3
                            + a ** 2 * a_conj ** 2 * (3 - 29 * b)
                            - (-1 + b) ** 3
                            + a * a_conj * (3 - 32 * b + 29 * b ** 2)
                        )
                        - 24
                        * a
                        * a_conj
                        * (
                            a ** 2 * a_conj ** 2
                            + a * a_conj * (2 - 3 * b)
                            + (-1 + b) ** 2
                        )
                        * b
                        * np.arctanh(
                            (
                                a ** 2 * a_conj ** 2
                                + (-1 + b) * b
                                + a * (a_conj + 2 * a_conj * b)
                            )
                            / (
                                np.sqrt((a * a_conj + b) ** 2)
                                * np.sqrt(
                                    a ** 2 * a_conj ** 2
                                    + (-1 + b) ** 2
                                    + 2 * a * a_conj * (1 + b)
                                )
                            )
                        )
                    )
                    + 84
                    * a
                    * a_conj
                    * b
                    * (a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b))
                    ** 3
                    * np.arctanh(
                        (
                            a ** 2 * a_conj ** 2
                            + b ** 2
                            - x ** 2
                            + a * a_conj * (1 + 2 * b - x ** 2)
                            + b * (-1 + x ** 2)
                        )
                        / (
                            np.sqrt(
                                a ** 2 * a_conj ** 2
                                + (-1 + b) ** 2
                                + 2 * a * a_conj * (1 + b)
                            )
                            * np.sqrt(
                                (a * a_conj + b) ** 2
                                - 2 * (a * a_conj - b) * x ** 2
                                + x ** 4
                            )
                        )
                    )
                    - 648
                    * a
                    * a_conj
                    * (1 + a * a_conj - b)
                    * b
                    * (a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b))
                    ** 2
                    * np.arctanh(
                        (
                            a ** 2 * a_conj ** 2
                            + b ** 2
                            - x ** 2
                            + a * a_conj * (1 + 2 * b - x ** 2)
                            + b * (-1 + x ** 2)
                        )
                        / (
                            np.sqrt(
                                a ** 2 * a_conj ** 2
                                + (-1 + b) ** 2
                                + 2 * a * a_conj * (1 + b)
                            )
                            * np.sqrt(
                                a ** 2 * a_conj ** 2
                                + 2 * a * a_conj * (b - x ** 2)
                                + (b + x ** 2) ** 2
                            )
                        )
                    )
                    - 960
                    * a
                    * a_conj
                    * b
                    * (
                        a ** 3 * a_conj ** 3
                        + 3 * a ** 2 * a_conj ** 2 * (1 - 2 * b)
                        - (-1 + b) ** 3
                        + 3 * a * a_conj * (1 - 3 * b + 2 * b ** 2)
                    )
                    * np.arctanh(
                        (
                            a ** 2 * a_conj ** 2
                            + b ** 2
                            - x ** 2
                            + a * a_conj * (1 + 2 * b - x ** 2)
                            + b * (-1 + x ** 2)
                        )
                        / (
                            np.sqrt(
                                a ** 2 * a_conj ** 2
                                + (-1 + b) ** 2
                                + 2 * a * a_conj * (1 + b)
                            )
                            * np.sqrt(
                                a ** 2 * a_conj ** 2
                                + 2 * a * a_conj * (b - x ** 2)
                                + (b + x ** 2) ** 2
                            )
                        )
                    )
                    - 1
                    / (1 + x ** 2) ** 2
                    * (
                        60
                        * (
                            a ** 2 * a_conj ** 2
                            + (-1 + b) ** 2
                            + 2 * a * a_conj * (1 + b)
                        )
                        * (
                            (
                                a ** 2 * a_conj ** 2
                                + a * a_conj * (2 - 8 * b)
                                + (-1 + b) ** 2
                            )
                            * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2)
                            ** (3 / 2)
                            * np.sqrt(
                                (a * a_conj + b) ** 2
                                - 2 * (a * a_conj - b) * x ** 2
                                + x ** 4
                            )
                            + np.sqrt(
                                1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2
                            )
                            * (
                                a ** 3 * a_conj ** 3
                                + a ** 2 * a_conj ** 2 * (3 - 29 * b)
                                - (-1 + b) ** 3
                                + a * a_conj * (3 - 32 * b + 29 * b ** 2)
                            )
                            * (1 + x ** 2)
                            * np.sqrt(
                                (a * a_conj + b) ** 2
                                - 2 * (a * a_conj - b) * x ** 2
                                + x ** 4
                            )
                            - 24
                            * a
                            * a_conj
                            * (
                                a ** 2 * a_conj ** 2
                                + a * a_conj * (2 - 3 * b)
                                + (-1 + b) ** 2
                            )
                            * b
                            * (1 + x ** 2) ** 2
                            * np.arctanh(
                                (
                                    a ** 2 * a_conj ** 2
                                    + b ** 2
                                    - x ** 2
                                    + a * a_conj * (1 + 2 * b - x ** 2)
                                    + b * (-1 + x ** 2)
                                )
                                / (
                                    np.sqrt(
                                        a ** 2 * a_conj ** 2
                                        + (-1 + b) ** 2
                                        + 2 * a * a_conj * (1 + b)
                                    )
                                    * np.sqrt(
                                        a ** 2 * a_conj ** 2
                                        + 2 * a * a_conj * (b - x ** 2)
                                        + (b + x ** 2) ** 2
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
        return -2 * np.pi * c * ret

    if abs(a_conj) < 0.00000001:
        ret = 0
        return ret


def zzzz_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return:
    """
    a_conj = np.conj(a)

    if abs(a_conj) >= 0.00000001:
        ret = (
            1
            / (
                12
                * b
                * (a * a_conj + b)
                * (a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b)) ** 4
            )
            * (
                np.sqrt((a * a_conj + b) ** 2)
                * (
                    3 * a ** 8 * a_conj ** 8
                    + 24 * a ** 7 * a_conj ** 7 * b
                    + 4 * a ** 6 * a_conj ** 6 * (-3 + 32 * b + 21 * b ** 2)
                    + 8 * a ** 5 * a_conj ** 5 * b * (-95 + 96 * b + 21 * b ** 2)
                    + 8
                    * a ** 3
                    * a_conj ** 3
                    * b
                    * (173 - 432 * b - 570 * b ** 2 + 320 * b ** 3 + 21 * b ** 4)
                    + 2
                    * a ** 4
                    * a_conj ** 4
                    * (9 - 232 * b - 1490 * b ** 2 + 960 * b ** 3 + 105 * b ** 4)
                    + (-1 + b) ** 3
                    * (
                        -3
                        + 39 * b
                        + 282 * b ** 2
                        + 342 * b ** 3
                        + 137 * b ** 4
                        + 3 * b ** 5
                    )
                    + 8
                    * a
                    * a_conj
                    * b
                    * (
                        -57
                        + 216 * b
                        + 261 * b ** 2
                        - 400 * b ** 3
                        - 119 * b ** 4
                        + 96 * b ** 5
                        + 3 * b ** 6
                    )
                    + 4
                    * a ** 2
                    * a_conj ** 2
                    * (
                        -3
                        + 144 * b
                        + 1035 * b ** 2
                        - 1464 * b ** 3
                        - 805 * b ** 4
                        + 480 * b ** 5
                        + 21 * b ** 6
                    )
                )
            )
            - 1
            / (
                12
                * (a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b)) ** 4
            )
            * (
                np.sqrt(
                    (a * a_conj + b) ** 2
                    - 2 * a * a_conj * x ** 2
                    + 2 * b * x ** 2
                    + x ** 4
                )
                * (
                    32
                    * (-1 + a * a_conj + b)
                    * (a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b))
                    ** 2
                    / (1 + x ** 2) ** 3
                    - 32
                    * (a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b))
                    * (
                        3 * a ** 3 * a_conj ** 3
                        + (-1 + b) ** 2 * (1 + 3 * b)
                        + a ** 2 * a_conj ** 2 * (-4 + 9 * b)
                        + 3 * a * a_conj * (-2 - 3 * b + 3 * b ** 2)
                    )
                    / (1 + x ** 2) ** 2
                    + 16
                    * (
                        9 * a ** 5 * a_conj ** 5
                        + 9 * a ** 4 * a_conj ** 4 * (-3 + 5 * b)
                        + (-1 + b) ** 3 * (5 + 12 * b + 9 * b ** 2)
                        + 2 * a ** 3 * a_conj ** 3 * (-23 - 48 * b + 45 * b ** 2)
                        + 2
                        * a ** 2
                        * a_conj ** 2
                        * (10 - 66 * b - 63 * b ** 2 + 45 * b ** 3)
                        + a
                        * a_conj
                        * (25 + 92 * b - 90 * b ** 2 - 72 * b ** 3 + 45 * b ** 4)
                    )
                    / (1 + x ** 2)
                    + 1
                    / (
                        b
                        * (
                            a ** 2 * a_conj ** 2
                            + 2 * a * a_conj * (b - x ** 2)
                            + (b + x ** 2) ** 2
                        )
                    )
                    * (
                        3
                        * (
                            a ** 9 * a_conj ** 9
                            + a ** 8 * a_conj ** 8 * (9 * b - x ** 2)
                            + 4
                            * a ** 7
                            * a_conj ** 7
                            * (-1 + 9 * b ** 2 - 2 * b * (-2 + x ** 2))
                            + 4
                            * a ** 6
                            * a_conj ** 6
                            * (
                                21 * b ** 3
                                + x ** 2
                                - 7 * b ** 2 * (-4 + x ** 2)
                                + b * (-31 + 4 * x ** 2)
                            )
                            + 2
                            * a ** 5
                            * a_conj ** 5
                            * (
                                3
                                + 63 * b ** 4
                                - 28 * b ** 3 * (-6 + x ** 2)
                                + 12 * b * (-2 + 5 * x ** 2)
                                + 6 * b ** 2 * (-47 + 8 * x ** 2)
                            )
                            + (-1 + b ** 2) ** 3
                            * (
                                b ** 3
                                + x ** 2
                                - b ** 2 * (-16 + x ** 2)
                                + b * (-1 + 16 * x ** 2)
                            )
                            + 4
                            * a ** 2
                            * a_conj ** 2
                            * (
                                9 * b ** 7
                                + x ** 2
                                + b ** 3 * (223 - 200 * x ** 2)
                                + b ** 2 * (100 - 169 * x ** 2)
                                - 7 * b ** 6 * (-12 + x ** 2)
                                + 3 * b * (-9 + 4 * x ** 2)
                                + 3 * b ** 5 * (-47 + 20 * x ** 2)
                                + 3 * b ** 4 * (-104 + 37 * x ** 2)
                            )
                            + 4
                            * a ** 3
                            * a_conj ** 3
                            * (
                                -1
                                + 21 * b ** 6
                                + b ** 2 * (223 - 112 * x ** 2)
                                - 14 * b ** 5 * (-10 + x ** 2)
                                - 6 * b * (-2 + 9 * x ** 2)
                                + 5 * b ** 4 * (-55 + 16 * x ** 2)
                                + 4 * b ** 3 * (-78 + 41 * x ** 2)
                            )
                            + a
                            * a_conj
                            * (-1 + b ** 2)
                            * (
                                -1
                                + 9 * b ** 6
                                + b ** 2 * (107 - 352 * x ** 2)
                                - 8 * b ** 5 * (-14 + x ** 2)
                                + 16 * b ** 3 * (-24 + 7 * x ** 2)
                                - 8 * b * (-2 + 13 * x ** 2)
                                + b ** 4 * (-115 + 96 * x ** 2)
                            )
                            + 2
                            * a ** 4
                            * a_conj ** 4
                            * (
                                63 * b ** 5
                                - 3 * x ** 2
                                - 35 * b ** 4 * (-8 + x ** 2)
                                - 3 * b * (-37 + 8 * x ** 2)
                                + 10 * b ** 3 * (-55 + 12 * x ** 2)
                                + b ** 2 * (-248 + 222 * x ** 2)
                            )
                        )
                    )
                )
            )
            - 4
            * (1 + a * a_conj + b)
            * (
                a ** 6 * a_conj ** 6
                + (-1 + b) ** 4 * (1 + b) ** 2
                + 2 * a ** 5 * a_conj ** 5 * (-4 + 3 * b)
                + a ** 4 * a_conj ** 4 * (-1 - 34 * b + 15 * b ** 2)
                + 2
                * a
                * a_conj
                * (-1 + b) ** 2
                * (-4 - 21 * b - 2 * b ** 2 + 3 * b ** 3)
                + 4 * a ** 3 * a_conj ** 3 * (4 - 7 * b - 14 * b ** 2 + 5 * b ** 3)
                + a ** 2
                * a_conj ** 2
                * (-1 + 108 * b - 54 * b ** 2 - 44 * b ** 3 + 15 * b ** 4)
            )
            * np.log(
                a * a_conj
                - b
                + (a * a_conj + b) ** 2
                + np.sqrt((a * a_conj + b) ** 2)
                * np.sqrt(1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2)
            )
            / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
            - 4
            * (1 + a * a_conj + b)
            * (
                a ** 6 * a_conj ** 6
                + (-1 + b) ** 4 * (1 + b) ** 2
                + 2 * a ** 5 * a_conj ** 5 * (-4 + 3 * b)
                + a ** 4 * a_conj ** 4 * (-1 - 34 * b + 15 * b ** 2)
                + 2
                * a
                * a_conj
                * (-1 + b) ** 2
                * (-4 - 21 * b - 2 * b ** 2 + 3 * b ** 3)
                + 4 * a ** 3 * a_conj ** 3 * (4 - 7 * b - 14 * b ** 2 + 5 * b ** 3)
                + a ** 2
                * a_conj ** 2
                * (-1 + 108 * b - 54 * b ** 2 - 44 * b ** 3 + 15 * b ** 4)
            )
            * np.log(1 + x ** 2)
            / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
            + 1
            / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
            * (
                4
                * (1 + a * a_conj + b)
                * (
                    a ** 6 * a_conj ** 6
                    + (-1 + b) ** 4 * (1 + b) ** 2
                    + 2 * a ** 5 * a_conj ** 5 * (-4 + 3 * b)
                    + a ** 4 * a_conj ** 4 * (-1 - 34 * b + 15 * b ** 2)
                    + 2
                    * a
                    * a_conj
                    * (-1 + b) ** 2
                    * (-4 - 21 * b - 2 * b ** 2 + 3 * b ** 3)
                    + 4 * a ** 3 * a_conj ** 3 * (4 - 7 * b - 14 * b ** 2 + 5 * b ** 3)
                    + a ** 2
                    * a_conj ** 2
                    * (-1 + 108 * b - 54 * b ** 2 - 44 * b ** 3 + 15 * b ** 4)
                )
                * np.log(
                    a ** 2 * a_conj ** 2
                    - b
                    + b ** 2
                    - x ** 2
                    + b * x ** 2
                    + a * a_conj * (1 + 2 * b - x ** 2)
                    + np.sqrt(
                        a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b)
                    )
                    * np.sqrt(
                        a ** 2 * a_conj ** 2
                        + 2 * a * a_conj * (b - x ** 2)
                        + (b + x ** 2) ** 2
                    )
                )
            )
        )
        return 4 * np.pi * c * ret
    if abs(a_conj) < 0.00000001 and (b - 1) ** 2 >= 0.00000001:
        ret = (
            16 * (-1 + b) ** 3
            - 48 * (-1 + b) ** 2 * b
            + 3 * (-1 + b) * (1 + b) ** 4 / b
            + 24 * (-1 + b) * (1 + 2 * b + 3 * b ** 2)
            - 16 * (-1 + b) ** 3 / (1 + x ** 2) ** 3
            + 48 * (-1 + b) ** 2 * b / (1 + x ** 2) ** 2
            - 24 * (-1 + b) * (1 + 2 * b + 3 * b ** 2) / (1 + x ** 2)
            - 3 * (-1 + b) * (1 + b) ** 4 / (b + x ** 2)
            - 24 * (1 + b) ** 3 * np.log(b)
            + 24 * (1 + b) ** 3 * np.log(b + x ** 2)
            - 24 * (1 + b) ** 3 * np.log(1 + x ** 2)
        ) / (6 * (-1 + b) ** 5)
        return 4 * c * np.pi * ret
    if abs(a_conj) < 0.00000001 and (b - 1) ** 2 < 0.00000001:
        ret = x ** 2 * (x ** 8 + 10 * x ** 4 + 5) / (10 * (x ** 2 + 1) ** 5)
        return 4 * c * np.pi * ret


def xxxz_integral(a, b, c, x):
    """
    function involved in computing the a_matrix (Rayleigh Ritz approx of the spectrum)
    @param a:
    @param b:
    @param c:
    @param x:
    @return:
    """
    a_conj = np.conj(a)

    if abs(a_conj) >= 0.00000001:
        ret = (
            1
            / 12
            * (a + a_conj)
            * (
                -(6 * (a ** 2 - a * a_conj + a_conj ** 2) / (a ** 3 * a_conj ** 3))
                - 3
                * (a ** 2 - a * a_conj + a_conj ** 2)
                * (-3 + a * a_conj + b)
                / (a ** 3 * a_conj ** 3)
                + 4
                * (a ** 2 - a * a_conj + a_conj ** 2)
                * (-1 + a * a_conj + b)
                / (a ** 3 * a_conj ** 3)
                - 1
                / (
                    a ** 3
                    * a_conj ** 3
                    * b
                    * (a ** 2 * a_conj ** 2 + (-1 + b) ** 2 + 2 * a * a_conj * (1 + b))
                    ** 4
                )
                * (
                    np.sqrt((a * a_conj + b) ** 2)
                    * (
                        a ** 10 * a_conj ** 8 * b
                        + a_conj ** 2 * (-3 + b) * (-1 + b) ** 6 * b ** 2
                        + a ** 9
                        * a_conj ** 7
                        * (3 - (-7 + a_conj ** 2) * b + 8 * b ** 2)
                        + a ** 8
                        * a_conj ** 6
                        * (
                            9
                            + 23 * b
                            + a_conj ** 4 * b
                            + 33 * b ** 2
                            + 28 * b ** 3
                            + a_conj ** 2 * (6 - 7 * b - 8 * b ** 2)
                        )
                        - a
                        * a_conj
                        * (-3 + b)
                        * (-1 + b) ** 4
                        * b
                        * ((-1 + b) ** 2 * b - a_conj ** 2 * (2 + 9 * b + 8 * b ** 2))
                        + a ** 7
                        * a_conj ** 5
                        * (
                            6
                            + 298 * b
                            + 13 * b ** 2
                            + 51 * b ** 3
                            + 56 * b ** 4
                            + a_conj ** 4 * (3 + 7 * b + 8 * b ** 2)
                            - a_conj ** 2 * (-18 + 119 * b + 33 * b ** 2 + 28 * b ** 3)
                        )
                        + a ** 6
                        * a_conj ** 4
                        * (
                            -6
                            + 473 * b
                            + 419 * b ** 2
                            - 85 * b ** 3
                            + 5 * b ** 4
                            + 70 * b ** 5
                            + a_conj ** 4 * (9 + 23 * b + 33 * b ** 2 + 28 * b ** 3)
                            + a_conj ** 2
                            * (12 + 113 * b - 478 * b ** 2 - 51 * b ** 3 - 56 * b ** 4)
                        )
                        + a ** 2
                        * (
                            (-3 + b) * (-1 + b) ** 6 * b ** 2
                            - a_conj ** 2
                            * (-1 + b) ** 4
                            * b
                            * (-15 - 34 * b - 15 * b ** 2 + 8 * b ** 3)
                            + a_conj ** 4
                            * (-1 + b) ** 2
                            * (
                                -3
                                - 95 * b
                                - 152 * b ** 2
                                - 89 * b ** 3
                                - 37 * b ** 4
                                + 28 * b ** 5
                            )
                        )
                        + a ** 5
                        * a_conj ** 3
                        * (
                            -9
                            + 109 * b
                            + 720 * b ** 2
                            + 82 * b ** 3
                            - 115 * b ** 4
                            - 75 * b ** 5
                            + 56 * b ** 6
                            + a_conj ** 4
                            * (6 + 298 * b + 13 * b ** 2 + 51 * b ** 3 + 56 * b ** 4)
                            - a_conj ** 2
                            * (
                                12
                                - 349 * b
                                + 35 * b ** 2
                                + 635 * b ** 3
                                + 5 * b ** 4
                                + 70 * b ** 5
                            )
                        )
                        + a ** 3
                        * (
                            a_conj
                            * (-1 + b) ** 4
                            * b
                            * (-6 - 25 * b - 15 * b ** 2 + 8 * b ** 3)
                            - a_conj ** 3
                            * (-1 + b) ** 2
                            * (
                                6
                                + 193 * b
                                + 451 * b ** 2
                                + 7 * b ** 3
                                - 37 * b ** 4
                                + 28 * b ** 5
                            )
                            + a_conj ** 5
                            * (
                                -9
                                + 109 * b
                                + 720 * b ** 2
                                + 82 * b ** 3
                                - 115 * b ** 4
                                - 75 * b ** 5
                                + 56 * b ** 6
                            )
                        )
                        + a ** 4
                        * (
                            a_conj ** 2
                            * (-1 + b) ** 2
                            * (
                                -3
                                - 95 * b
                                - 152 * b ** 2
                                - 89 * b ** 3
                                - 37 * b ** 4
                                + 28 * b ** 5
                            )
                            + a_conj ** 6
                            * (
                                -6
                                + 473 * b
                                + 419 * b ** 2
                                - 85 * b ** 3
                                + 5 * b ** 4
                                + 70 * b ** 5
                            )
                            - a_conj ** 4
                            * (
                                18
                                + 73 * b
                                - 888 * b ** 2
                                + 466 * b ** 3
                                + 350 * b ** 4
                                - 75 * b ** 5
                                + 56 * b ** 6
                            )
                        )
                    )
                )
                - 4
                * (a ** 2 - a * a_conj + a_conj ** 2)
                * (-1 + a * a_conj + b)
                / (a ** 3 * a_conj ** 3 * (1 + x ** 2) ** 3)
                + 3
                * (a ** 2 - a * a_conj + a_conj ** 2)
                * (-3 + a * a_conj + b)
                / (a ** 3 * a_conj ** 3 * (1 + x ** 2) ** 2)
                + 6
                * (a ** 2 - a * a_conj + a_conj ** 2)
                / (a ** 3 * a_conj ** 3 * (1 + x ** 2))
                + 1
                / (
                    a ** 3
                    * a_conj ** 3
                    * (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** 4
                )
                * (
                    np.sqrt(
                        (a * a_conj + b) ** 2
                        - 2 * a * a_conj * x ** 2
                        + 2 * b * x ** 2
                        + x ** 4
                    )
                    * (
                        1
                        / (1 + x ** 2) ** 3
                        * (
                            4
                            * (
                                a ** 2 * a_conj ** 2
                                + (-1 + b) ** 2
                                + 2 * a * a_conj * (1 + b)
                            )
                            ** 2
                            * (
                                a ** 6 * a_conj ** 4
                                + a_conj ** 2 * (-1 + b) ** 4
                                + a ** 5 * a_conj ** 3 * (2 - a_conj ** 2 + 4 * b)
                                + a ** 4
                                * (
                                    a_conj ** 6
                                    + 6 * a_conj ** 2 * b ** 2
                                    - 2 * a_conj ** 4 * (1 + 2 * b)
                                )
                                + a ** 2
                                * (
                                    (-1 + b) ** 4
                                    + 6 * a_conj ** 4 * b ** 2
                                    - 2 * a_conj ** 2 * (-1 + b) ** 2 * (1 + 2 * b)
                                )
                                + a
                                * (
                                    -a_conj * (-1 + b) ** 4
                                    + a_conj ** 3 * (-1 + b) ** 2 * (2 + 4 * b)
                                )
                                + 2
                                * a ** 3
                                * (
                                    a_conj ** 5 * (1 + 2 * b)
                                    + a_conj * (-1 + b) ** 2 * (1 + 2 * b)
                                    - 3 * a_conj ** 3 * (1 + b ** 2)
                                )
                            )
                        )
                        - 1
                        / (1 + x ** 2) ** 2
                        * (
                            (
                                a ** 2 * a_conj ** 2
                                + (-1 + b) ** 2
                                + 2 * a * a_conj * (1 + b)
                            )
                            * (
                                3 * a ** 8 * a_conj ** 6
                                + a_conj ** 2 * (-1 + b) ** 5 * (-5 + 3 * b)
                                + a ** 7 * a_conj ** 5 * (2 - 3 * a_conj ** 2 + 18 * b)
                                + a
                                * (
                                    -a_conj * (-1 + b) ** 5 * (-5 + 3 * b)
                                    + 6
                                    * a_conj ** 3
                                    * (-1 + b) ** 3
                                    * (-5 - 4 * b + 3 * b ** 2)
                                )
                                + a ** 6
                                * (
                                    3 * a_conj ** 8
                                    - 2 * a_conj ** 6 * (1 + 9 * b)
                                    + 3 * a_conj ** 4 * (-9 - 4 * b + 15 * b ** 2)
                                )
                                + a ** 5
                                * a_conj ** 3
                                * (
                                    16
                                    - 68 * b
                                    - 68 * b ** 2
                                    + 60 * b ** 3
                                    + 2 * a_conj ** 4 * (1 + 9 * b)
                                    - 3 * a_conj ** 2 * (21 - 4 * b + 15 * b ** 2)
                                )
                                + a ** 4
                                * a_conj ** 2
                                * (
                                    67
                                    - 112 * b ** 3
                                    + 45 * b ** 4
                                    + 3 * a_conj ** 4 * (-9 - 4 * b + 15 * b ** 2)
                                    - 4
                                    * a_conj ** 2
                                    * (16 + 28 * b - 17 * b ** 2 + 15 * b ** 3)
                                )
                                + a ** 3
                                * (
                                    6
                                    * a_conj
                                    * (-1 + b) ** 3
                                    * (-5 - 4 * b + 3 * b ** 2)
                                    + 4
                                    * a_conj ** 5
                                    * (4 - 17 * b - 17 * b ** 2 + 15 * b ** 3)
                                    + a_conj ** 3
                                    * (
                                        -25
                                        + 48 * b
                                        - 90 * b ** 2
                                        + 112 * b ** 3
                                        - 45 * b ** 4
                                    )
                                )
                                + a ** 2
                                * (
                                    (-1 + b) ** 5 * (-5 + 3 * b)
                                    - 6
                                    * a_conj ** 2
                                    * (-1 + b) ** 3
                                    * (-5 - 4 * b + 3 * b ** 2)
                                    + a_conj ** 4 * (67 - 112 * b ** 3 + 45 * b ** 4)
                                )
                            )
                        )
                        + 1
                        / (1 + x ** 2)
                        * (
                            -9 * a ** 9 * a_conj ** 7
                            + a ** 8 * a_conj ** 6 * (-47 + 9 * a_conj ** 2 - 57 * b)
                            + a * a_conj * (-1 + b) ** 6 * (-1 + 3 * b)
                            - a_conj ** 2 * (-1 + b) ** 6 * (-1 + 3 * b)
                            - a
                            * a_conj ** 3
                            * (-1 + b) ** 4
                            * (-7 + 24 * b + 27 * b ** 2)
                            - a ** 7
                            * a_conj ** 5
                            * (
                                -115
                                + 9 * a_conj ** 4
                                + a_conj ** 2 * (97 - 57 * b)
                                + 180 * b
                                + 153 * b ** 2
                            )
                            - a ** 6
                            * a_conj ** 4
                            * (
                                -353
                                - 259 * b
                                + 231 * b ** 2
                                + 225 * b ** 3
                                + a_conj ** 4 * (47 + 57 * b)
                                + a_conj ** 2 * (61 + 396 * b - 153 * b ** 2)
                            )
                            + a ** 2
                            * (
                                -((-1 + b) ** 6) * (-1 + 3 * b)
                                + a_conj ** 2
                                * (-1 + b) ** 4
                                * (-7 + 24 * b + 27 * b ** 2)
                                - a_conj ** 4
                                * (-1 + b) ** 2
                                * (-29 + 35 * b + 99 * b ** 2 + 99 * b ** 3)
                            )
                            + a ** 5
                            * a_conj ** 3
                            * (
                                223
                                + 492 * b
                                + 216 * b ** 2
                                - 64 * b ** 3
                                - 195 * b ** 4
                                + a_conj ** 4 * (115 - 180 * b - 153 * b ** 2)
                                + a_conj ** 2
                                * (109 - 205 * b - 633 * b ** 2 + 225 * b ** 3)
                            )
                            + a ** 4
                            * a_conj ** 2
                            * (
                                29
                                - 93 * b
                                + 64 * b ** 3
                                + 99 * b ** 4
                                - 99 * b ** 5
                                + a_conj ** 4
                                * (353 + 259 * b - 231 * b ** 2 - 225 * b ** 3)
                                + a_conj ** 2
                                * (
                                    -37
                                    + 432 * b
                                    - 270 * b ** 2
                                    - 512 * b ** 3
                                    + 195 * b ** 4
                                )
                            )
                            - a ** 3
                            * (
                                a_conj * (-1 + b) ** 4 * (-7 + 24 * b + 27 * b ** 2)
                                - a_conj ** 3
                                * (-1 + b) ** 2
                                * (-107 - 307 * b - 45 * b ** 2 + 99 * b ** 3)
                                + a_conj ** 5
                                * (
                                    -223
                                    - 492 * b
                                    - 216 * b ** 2
                                    + 64 * b ** 3
                                    + 195 * b ** 4
                                )
                            )
                        )
                        + 1
                        / (
                            b
                            * (
                                a ** 2 * a_conj ** 2
                                + 2 * a * a_conj * (b - x ** 2)
                                + (b + x ** 2) ** 2
                            )
                        )
                        * (
                            3
                            * a
                            * a_conj
                            * (a + a_conj) ** 2
                            * (
                                a ** 8 * a_conj ** 8
                                - a ** 7 * a_conj ** 7 * (-3 + x ** 2)
                                + (-1 + b) ** 4 * b ** 2 * (1 + b) * (b + x ** 2)
                                + a ** 6
                                * a_conj ** 6
                                * (2 - 20 * b ** 2 - 3 * x ** 2 + b * (31 - 5 * x ** 2))
                                + a
                                * a_conj
                                * (-1 + b) ** 2
                                * b
                                * (
                                    b
                                    - 6 * x ** 2
                                    - 31 * b * x ** 2
                                    + b ** 3 * (-31 + 5 * x ** 2)
                                    - 2 * b ** 2 * (9 + 8 * x ** 2)
                                )
                                - a ** 5
                                * a_conj ** 5
                                * (
                                    64 * b ** 3
                                    + 2 * (1 + x ** 2)
                                    + b ** 2 * (-75 + 9 * x ** 2)
                                    + b * (-44 + 26 * x ** 2)
                                )
                                - a ** 3
                                * a_conj ** 3
                                * (
                                    1
                                    + 64 * b ** 5
                                    - 3 * x ** 2
                                    + b * (20 - 40 * x ** 2)
                                    + b ** 4 * (47 - 5 * x ** 2)
                                    + 4 * b ** 2 * (1 + x ** 2)
                                    + 4 * b ** 3 * (-58 + 27 * x ** 2)
                                )
                                + a ** 2
                                * a_conj ** 2
                                * (
                                    -20 * b ** 6
                                    + x ** 2
                                    + b ** 4 * (158 - 77 * x ** 2)
                                    + 4 * b ** 3 * (1 + x ** 2)
                                    + b ** 5 * (-75 + 9 * x ** 2)
                                    + b * (-1 + 19 * x ** 2)
                                    + b ** 2 * (-34 + 76 * x ** 2)
                                )
                                - a ** 4
                                * a_conj ** 4
                                * (
                                    3
                                    + 90 * b ** 4
                                    - 2 * x ** 2
                                    + 6 * b * (1 + x ** 2)
                                    + b ** 3 * (-47 + 5 * x ** 2)
                                    + b ** 2 * (-158 + 77 * x ** 2)
                                )
                            )
                        )
                    )
                )
                - 1
                / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
                * (
                    12
                    * (-1 + a * a_conj + b)
                    * (1 + a * a_conj + b)
                    * (
                        a ** 4 * a_conj ** 2 * (-10 + 3 * a_conj ** 2)
                        - 10 * a_conj ** 2 * (1 + 5 * b + b ** 2)
                        + 3 * (-1 + b) ** 2 * (1 + 8 * b + b ** 2)
                        - 4
                        * a
                        * a_conj
                        * (
                            2
                            + 13 * b
                            - 7 * b ** 2
                            - 3 * b ** 3
                            + 5 * a_conj ** 2 * (1 + b)
                        )
                        + 4
                        * a ** 3
                        * (-5 * a_conj * (1 + b) + a_conj ** 3 * (-2 + 3 * b))
                        - 2
                        * a ** 2
                        * (
                            5 * a_conj ** 4
                            + 5 * (1 + 5 * b + b ** 2)
                            - a_conj ** 2 * (-11 + b + 9 * b ** 2)
                        )
                    )
                    * np.log(
                        a * a_conj
                        - b
                        + (a * a_conj + b) ** 2
                        + np.sqrt((a * a_conj + b) ** 2)
                        * np.sqrt(1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2)
                    )
                )
                - 1
                / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
                * (
                    12
                    * (-1 + a * a_conj + b)
                    * (1 + a * a_conj + b)
                    * (
                        a ** 4 * a_conj ** 2 * (-10 + 3 * a_conj ** 2)
                        - 10 * a_conj ** 2 * (1 + 5 * b + b ** 2)
                        + 3 * (-1 + b) ** 2 * (1 + 8 * b + b ** 2)
                        - 4
                        * a
                        * a_conj
                        * (
                            2
                            + 13 * b
                            - 7 * b ** 2
                            - 3 * b ** 3
                            + 5 * a_conj ** 2 * (1 + b)
                        )
                        + 4
                        * a ** 3
                        * (-5 * a_conj * (1 + b) + a_conj ** 3 * (-2 + 3 * b))
                        - 2
                        * a ** 2
                        * (
                            5 * a_conj ** 4
                            + 5 * (1 + 5 * b + b ** 2)
                            - a_conj ** 2 * (-11 + b + 9 * b ** 2)
                        )
                    )
                    * np.log(1 + x ** 2)
                )
                + 1
                / (1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2) ** (9 / 2)
                * (
                    12
                    * (-1 + a * a_conj + b)
                    * (1 + a * a_conj + b)
                    * (
                        a ** 4 * a_conj ** 2 * (-10 + 3 * a_conj ** 2)
                        - 10 * a_conj ** 2 * (1 + 5 * b + b ** 2)
                        + 3 * (-1 + b) ** 2 * (1 + 8 * b + b ** 2)
                        - 4
                        * a
                        * a_conj
                        * (
                            2
                            + 13 * b
                            - 7 * b ** 2
                            - 3 * b ** 3
                            + 5 * a_conj ** 2 * (1 + b)
                        )
                        + 4
                        * a ** 3
                        * (-5 * a_conj * (1 + b) + a_conj ** 3 * (-2 + 3 * b))
                        - 2
                        * a ** 2
                        * (
                            5 * a_conj ** 4
                            + 5 * (1 + 5 * b + b ** 2)
                            - a_conj ** 2 * (-11 + b + 9 * b ** 2)
                        )
                    )
                    * np.log(
                        a * a_conj
                        - b
                        + (a * a_conj + b) ** 2
                        - x ** 2
                        - a * a_conj * x ** 2
                        + b * x ** 2
                        + np.sqrt(1 + 2 * a * a_conj - 2 * b + (a * a_conj + b) ** 2)
                        * np.sqrt(
                            a ** 2 * a_conj ** 2
                            + 2 * a * a_conj * (b - x ** 2)
                            + (b + x ** 2) ** 2
                        )
                    )
                )
            )
        )
        return 4 * np.pi * c * ret
    if abs(a_conj) < 0.00000001:
        return 0
