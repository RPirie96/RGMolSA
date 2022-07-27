"""
script to parameterize RDKit molecule object for descriptor calculation.

Functions:
- param_mol: script to convert molecule into constructs

Exceptions:
- TypeError: raised if RDKit cannot generate 3D coordinates for molecule
- AttributeError: raised if base sphere not correctly assigned post ring-replacement
- ArithmeticError: Raised if Negative Surface Area computed
"""

from collections import namedtuple
import numpy as np

from scipy.spatial import distance_matrix
from numpy import linalg as la
from rdkit import Chem

from conf_gen import embed_3d
from utils import get_chain


def get_levels(adjacency_matrix, no_atoms, base_sphere):

    """
    Helper function to generate matrix of levels starting from base sphere. produce a matrix of integers row = level;
    1 = non-terminal at this level, 2 = terminal at this level

    @param adjacency_matrix:
    @param no_atoms:
    @param base_sphere:
    @return: level_mat, no_levels
    """

    r_sum = adjacency_matrix.sum(axis=1)
    to_do = no_atoms - 1  # how may remaining spheres need to be assigned
    assigned, level_mat = np.zeros((1, no_atoms), dtype=int), np.zeros(
        (1, no_atoms), dtype=int
    )
    assigned[0, base_sphere] = 1
    level_mat[0, base_sphere] = 1

    current_level = 0
    while to_do > 0 and current_level < 500:
        next_level = np.zeros((1, no_atoms), dtype=int)

        for j in range(0, no_atoms):
            if level_mat[current_level, j] == 1:
                current_sphere = j

                for i in range(0, no_atoms):
                    if (
                        adjacency_matrix[current_sphere, i] == 1
                        and r_sum[i] == 1
                        and assigned[0, i] == 0
                    ):
                        next_level[0, i] = 2
                        assigned[0, i] = 1
                        to_do += -1
                    if (
                        adjacency_matrix[current_sphere, i] == 1
                        and r_sum[i] > 1
                        and assigned[0, i] == 0
                    ):
                        next_level[0, i] = 1
                        assigned[0, i] = 1
                        to_do += -1

        level_mat = np.vstack((level_mat, next_level))
        current_level += 1

    no_levels = len(level_mat) - 1  # number of levels

    return level_mat, no_levels


def get_area(adjacency_matrix, centres, no_atoms, radii):

    """
    Helper function to return the surface area of the molecule, and the matrix of lambda values

    If the area is negative (usually for bridged bicyclic compounds with >2 intersecting rings) a
    ValueError is raised. As the area is computed as the area of a sphere - the bit where two spheres
    intersect, multiple large spheres intersecting leads to a negative value, and thus the surface of the
    molecule cannot be approximated.

    @param adjacency_matrix:
    @param centres:
    @param no_atoms:
    @param radii:
    @return: area and matrix of lambda values
    """

    # matrix of distances between intersecting atoms
    distances = adjacency_matrix * distance_matrix(centres, centres)

    # matrix of lambdas
    lam = np.zeros((no_atoms, no_atoms))
    for i in range(0, no_atoms):
        for j in range(0, no_atoms):
            if adjacency_matrix[i, j] == 1:
                lam[i, j] = (radii[i] ** 2 - radii[j] ** 2 + distances[i, j] ** 2) / (
                    2 * distances[i, j]
                )
            else:
                lam[i, j] = 0

    # surface area of the molecule
    area = 0
    for i in range(0, no_atoms):
        sphere_i = 4 * np.pi * radii[i] ** 2
        for j in range(0, no_atoms):
            if adjacency_matrix[i, j] == 1:
                sphere_i = sphere_i - 2 * radii[i] * np.pi * abs(radii[i] - lam[i, j])
        area += sphere_i

    if area < 0:
        raise ArithmeticError("Negative Surface Area, cannot approximate surface")

    return lam, area


def rescale_inputs(area, centres, radii, lam):

    """
    Helper function to rescale all inputs to give total surface area equal to 4pi

    @param area:
    @param centres:
    @param radii:
    @param lam:
    @return: inputs rescaled to have surface area 4pi
    """

    centres_r = centres * np.sqrt(4 * np.pi / area)
    radii_r = radii * np.sqrt(4 * np.pi / area)
    lam_r = lam * np.sqrt(4 * np.pi / area)

    return centres_r, radii_r, lam_r


def base_error(no_levels, level_mat, no_atoms, adjacency_matrix, base_sphere, centres_r, radii_r, lam_r):
    """
    Helper function to return the vector of next level spheres and the updated rescaled centres post-error handling

    @param no_levels:
    @param level_mat:
    @param no_atoms:
    @param adjacency_matrix:
    @param base_sphere:
    @param centres_r:
    @param radii_r:
    @param lam_r:
    @return: updated centres
    """
    # Fingerprint Matrix that tells you how to navigate through molecule
    fingerprint = np.tile(-1, (no_levels + 1, no_atoms))
    for i in range(0, no_levels + 1):
        for j in range(0, no_atoms):
            if level_mat[i][j] > 0:
                s_list = get_chain(no_atoms, level_mat, adjacency_matrix, j, i)
                for k in range(0, len(s_list)):
                    fingerprint[k][j] = s_list[k]

    # Code to produce vector of next level spheres
    sphere_levels_vec = []
    next_level = []
    for i in range(0, no_atoms):
        stop = 0
        j = 0
        while stop < 1:
            if fingerprint[j][i] == i:
                stop = 1
            else:
                j = j + 1
        sphere_levels_vec.append(j)
        next_spheres = []
        for s_n in range(0, no_atoms):
            if (
                j < no_levels
                and fingerprint[j][s_n] == i
                and fingerprint[j + 1][s_n] == s_n
            ):
                next_spheres.append(s_n)
        next_level.append(next_spheres)

    # Error handling code - take the base sphere and rotate so that north pole is in base sphere
    fine = 0
    cover_sphere = base_sphere
    i = 0
    while i < len(next_level[base_sphere]) and fine == 0:
        check_sphere = next_level[base_sphere][i]
        if (
            la.norm(
                centres_r[check_sphere] - radii_r[base_sphere] * np.array([0, 0, 1])
            )
            <= radii_r[check_sphere]
        ):
            cover_sphere = check_sphere
            fine = 1  # if there is something over the north pole
        i += 1

    fine_2 = 0
    angle_x = 10
    while angle_x <= np.pi and fine_2 == 0:

        # define matrix to rotate about x and y
        rot_mat_x = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_x), -np.sin(angle_x)],
                [0, np.sin(angle_x), np.cos(angle_x)],
            ]
        )
        centres_r = np.matmul(centres_r, rot_mat_x)

        unit_cover = (1 / la.norm(centres_r[cover_sphere])) * centres_r[cover_sphere]
        plane_point = 0.85 * lam_r[base_sphere][cover_sphere] * unit_cover
        v_rand = np.random.rand(3)
        v_rand = v_rand / (la.norm(v_rand))
        w_rand = np.cross(unit_cover, v_rand)

        a_coefficient = la.norm(w_rand) ** 2
        b_coefficient = 2 * np.dot(plane_point, w_rand)
        c_coefficient = la.norm(plane_point) ** 2 - radii_r[base_sphere] ** 2

        mu = (
            -b_coefficient
            + np.sqrt(b_coefficient ** 2 - 4 * a_coefficient * c_coefficient)
        ) / (2 * a_coefficient)
        test_point = plane_point + mu * w_rand
        fine_2 = 1
        for i in range(0, len(next_level[base_sphere])):
            check_sphere = next_level[base_sphere][i]
            if la.norm(centres_r[check_sphere] - test_point) <= radii_r[check_sphere]:
                fine_2 = 0

        angle_x = angle_x + 10

    return sphere_levels_vec, next_level, centres_r


def param_mol(mol):
    """
    Function to convert RDKit Molecule object into the required inputs for calculation of RGMolSA descriptor
    @param mol:
    @return:
    """

    # check for 3D coords
    if "3D" not in Chem.MolToMolBlock(mol).split("\n")[1]:
        mol = embed_3d(mol)

    # check that the above embed has worked
    if "3D" not in Chem.MolToMolBlock(mol).split("\n")[1]:
        raise TypeError("RDKit cannot produce 3D coordinates for this molecule")

    # number of atoms in molecule
    no_atoms = mol.GetNumAtoms()

    # Atomic centres
    pos = [mol.GetConformer().GetAtomPosition(i) for i in range(0, no_atoms)]
    centres = np.array([[p.x, p.y, p.z] for p in pos])

    # Atomic radii
    radii = np.array(
        [
            0.6 * Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum())
            for a in mol.GetAtoms()
        ]
    )

    # Adjacency matrix
    adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)

    # get initial base sphere
    # Find the centroid
    centroid = [
        np.sum(centres[:, 0]) / len(centres[:, 0]),
        np.sum(centres[:, 1]) / len(centres[:, 1]),
        np.sum(centres[:, 2]) / len(centres[:, 2]),
    ]

    # Find the index of the minimum Euclidean distance and set this as the base sphere
    base_sphere = np.argmin(np.sqrt(np.sum(np.square(centres - centroid), axis=1)))

    # get the number of rings in the molecule
    num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)

    # if the molecule contains rings
    if num_rings != 0:
        # list of all atom indices
        indices = [a.GetIdx() for a in mol.GetAtoms()]

        # list of ring indices
        rings = [list(mol.GetRingInfo().AtomRings()[i]) for i in range(num_rings)]

        ring_pos = list(
            set(sorted([index for sublist in rings for index in sublist]))
        )  # this creates a sorted list of ring indexes
        non_ring = [x for x in indices if x not in ring_pos]
        updated_non_ring = [i + len(rings) for i in range(len(non_ring))]

        # update base sphere
        base_found = False
        for i in range(len(rings)):
            print(base_sphere)
            if base_sphere in rings[i]:
                base_sphere = i
                base_found = True

        # if base sphere not in ring
        if not base_found:
            base_sphere = updated_non_ring[non_ring.index(base_sphere)]
            base_found = True

        # raise AttributeError if new position of base sphere not found
        if not base_found:
            raise AttributeError("Base Sphere not assigned")

        # update molecule information structures to account for rings

        # get a list of the atom centres contained in the ring
        ring_centres = [np.take(centres, ring, axis=0) for ring in rings]

        # get centroid of each ring
        centroids = np.array(
            [
                [
                    np.sum(ring[:, 0]) / len(ring[:, 0]),
                    np.sum(ring[:, 1]) / len(ring[:, 1]),
                    np.sum(ring[:, 2]) / len(ring[:, 2]),
                ]
                for ring in ring_centres
            ]
        )

        # remove ring atoms and add in spheres at the beginning of each structure
        # choose beginning as easier to index using num_rings!

        no_atoms = (
                no_atoms - len(ring_pos) + len(rings)
        )  # update inputs to reflect ring --> sphere
        centres = np.vstack(
            (centroids, np.delete(centres, ring_pos, axis=0))
        )  # remove ring atom centres and add centroids
        radii = np.append(
            np.full(len(rings), 2.25), np.delete(radii, ring_pos)
        )  # remove ring atom radii and add ring

        new_am = np.zeros(
            (no_atoms, no_atoms), dtype=int
        )  # define empty matrix of new size

        # intersection between rings
        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                for k in rings[i]:
                    for l in rings[j]:
                        if adjacency_matrix[k][l] == 1 and new_am[i][j] == 0:
                            new_am[i][j] = 1
                            new_am[j][i] = 1

            # intersection between ring and non ring
            for k in rings[i]:
                for l in range(len(non_ring)):
                    if (
                            adjacency_matrix[k][non_ring[l]] == 1
                            and new_am[i][updated_non_ring[l]] == 0
                    ):
                        new_am[i][updated_non_ring[l]] = 1
                        new_am[updated_non_ring[l]][i] = 1

        # intersection between non ring
        for i in range(len(non_ring)):
            for j in range(len(non_ring)):
                if (
                        adjacency_matrix[non_ring[i]][non_ring[j]] == 1
                        and new_am[updated_non_ring[i]][updated_non_ring[j]] == 0
                ):
                    new_am[updated_non_ring[i]][updated_non_ring[j]] = 1
                    new_am[updated_non_ring[j]][updated_non_ring[i]] = 1

    else:
        new_am = adjacency_matrix

    # re-centre so the base sphere site on the origin
    c_rel = centres - centres[base_sphere]
    centres = c_rel[:]

    # get levels within molecule
    level_mat, no_levels = get_levels(new_am, no_atoms, base_sphere)

    # get molecule area
    lam, area = get_area(new_am, centres, no_atoms, radii)

    # rescale inputs so molecule has surface area equivalent to a unit sphere
    centres_r, radii_r, lam_r = rescale_inputs(area, centres, radii, lam)

    sphere_levels_vec, next_level, centres_r = base_error(no_levels, level_mat, no_atoms, new_am, base_sphere,
                                                          centres_r, radii_r, lam_r)

    inputs = namedtuple("inputs", ["no_atoms", "adjacency_matrix", "radii_r", "lam_r", "centres_r", "level_mat", "no_levels",
                                   "sphere_levels_vec", "next_level"])

    return inputs(
        no_atoms=no_atoms, adjacency_matrix=new_am, area=area, radii_r=radii_r, lam_r=lam_r, centres_r=centres_r,
        level_mat=level_mat, no_levels=no_levels, sphere_levels_vec=sphere_levels_vec, next_level=next_level
    )
