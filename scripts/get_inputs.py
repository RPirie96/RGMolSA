# script to get initial inputs
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import itertools
from collections import namedtuple
import copy


def get_mol_info(mol):
    """
        Function which reads in ROMol structure, replaces rings with spheres and outputs no atoms, 3D centres,
        adjacency matrix and list of radii

        :param mol:
        :return adjacency_matrix, no_atoms, centres, radii:
    """

    # number of atoms in molecule
    no_atoms = mol.GetNumAtoms()

    # Atomic centres
    pos = [mol.GetConformer().GetAtomPosition(i) for i in range(0, no_atoms)]
    centres = np.array([[p.x, p.y, p.z] for p in pos])

    # Atomic radii
    radii = np.array([0.6*Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()])

    # Adjacency matrix
    adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)

    # get the number of rings in the molecule
    num_rings = rdMolDescriptors.CalcNumRings(mol)

    # if the molecule contains rings
    if num_rings != 0:

        # get list of ring atom indices
        rings = [list(mol.GetRingInfo().AtomRings()[i]) for i in range(num_rings)]

        # fused and bicyclic rings, non-ring neighbours
        fused, bicyclic, nrns, bonded_rings, bonded = [], [], [], [], []
        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                count = 0
                for idx in rings[i]:
                    if idx in rings[j]:
                        count += 1
                        fused.append([i, j])  # position of fused rings
                    for neighbour in mol.GetAtomWithIdx(idx).GetNeighbors():
                        if neighbour.GetIdx() in rings[j] and neighbour.GetIdx() not in rings[i] and idx not in rings[
                            j]:
                            bonded_rings.append([i, j])
                            bonded.append(neighbour.GetIdx())

                if count >= 3:
                    bicyclic.append([i, j])  # position of bicyclic rings

            non_ring_neighbours = []
            for idx in rings[i]:
                for neighbour in mol.GetAtomWithIdx(idx).GetNeighbors():
                    if not any(neighbour.GetIdx() in ring for ring in rings):
                        non_ring_neighbours.append(neighbour.GetIdx())  # position of non ring neighbours

            nrns.append(non_ring_neighbours)

        # remove duplicates from fused and fused from single bonded
        fused.sort()
        fused = list(i for i, _ in itertools.groupby(fused))

        not_bonded = []
        for i in range(len(bonded_rings)):
            n = bonded[i]
            br = bonded_rings[i]
            for j in range(len(rings)):
                if n in rings[j] and j != br[0] and j != br[1]:
                    not_bonded.append(br)

        bonded_rings = [x for x in bonded_rings if x not in not_bonded]

        # group together multiple intersecting spheres
        grouped = {}
        for name, x in bicyclic:
            grouped.setdefault(name, []).append(x)
        keys = list(grouped.keys())

        if bicyclic:
            bicyclic_done = np.full(len(rings), False, dtype=bool)

            rings_old = len(copy.deepcopy(rings))  # keep copy of old rings list length to update indexes

            for i in keys:
                if not bicyclic_done[i]:
                    for j in grouped[i]:
                        if not bicyclic_done[j]:
                            # ring indices
                            bidx = rings[i] + rings[j]
                            bidx.sort()
                            bidx = list(i for i, _ in itertools.groupby(bidx))
                            rings[i] = bidx

                            # neighbour indexes
                            bnrns = nrns[i] + nrns[j]
                            bnrns.sort()
                            bnrns = list(i for i, _ in itertools.groupby(bnrns))
                            nrns[i] = bnrns

                            bicyclic_done[i] = True
                            bicyclic_done[j] = True
                    for index in sorted(grouped[i], reverse=True):
                        del rings[index]
                        del nrns[index]

            idx_factor = rings_old - len(rings)

            # update list of fused ring atom indexes to account for ring fused to bicyclic
            fused = []
            for i in range(len(rings)):
                for j in range(i + 1, len(rings)):
                    for idx in rings[i]:
                        if idx in rings[j]:
                            fused.append([i, j])  # position of shared atoms

            # remove duplicates from list
            fused.sort()
            fused = list(i for i, _ in itertools.groupby(fused))

            # update bonded ring positions
            bonded_rings = [[br[0] - idx_factor, br[1] - idx_factor] for br in bonded_rings]

        # update molecule information structures to account for rings

        # get a list of the atom centres contained in the ring
        ring_centres = [np.take(centres, ring, axis=0) for ring in rings]

        # get centroid of each ring
        centroids = np.array([[np.sum(ring[:, 0]) / len(ring[:, 0]), np.sum(ring[:, 1]) / len(ring[:, 1]),
                               np.sum(ring[:, 2]) / len(ring[:, 2])] for ring in ring_centres])

        # get a list of the atom centres that neighbour the ring
        neighbour_centres = [np.take(centres, n, axis=0) for n in nrns]

        # remove ring atoms and add in spheres at the beginning of each structure
        # choose beginning as easier to index using num_rings!
        ring_pos = list(set(sorted(
            [index for sublist in rings for index in sublist])))  # this creates a sorted list of ring indexes

        no_atoms = no_atoms - len(ring_pos) + len(rings)  # update inputs to reflect ring --> sphere
        centres = np.vstack(
            (centroids, np.delete(centres, ring_pos, axis=0)))  # remove ring atom centres and add centroids
        radii = np.append(np.full(len(rings), 2.25), np.delete(radii, ring_pos))  # remove ring atom radii and add ring

        # get list of updated neighbour positions
        updated_nrns = []
        for n in neighbour_centres:
            ids = []
            for pos in n:
                for i in range(len(centres)):
                    if centres[i][0] == pos[0] and centres[i][1] == pos[1] and centres[i][2] == pos[2]:
                        ids.append(i)
            updated_nrns.append(ids)

        adjacency_matrix = np.delete(adjacency_matrix, ring_pos, axis=0)
        adjacency_matrix = np.delete(adjacency_matrix, ring_pos, axis=1)

        # set up adjacency matrix with rows + columns of zeros
        adjacency_matrix = np.vstack((np.zeros((len(rings), (no_atoms - len(rings))), dtype=int), adjacency_matrix))
        adjacency_matrix = np.hstack((np.zeros((no_atoms, len(rings)), dtype=int), adjacency_matrix))

        # intersection of sphere with nrns
        for i in range(len(rings)):
            for j in updated_nrns[i]:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

        # intersection between fused spheres
        for i in fused:
            adjacency_matrix[i[0]][i[1]] = 1
            adjacency_matrix[i[1]][i[0]] = 1

        for i in bonded_rings:
            adjacency_matrix[i[0]][i[1]] = 1
            adjacency_matrix[i[1]][i[0]] = 1

    inputs = namedtuple('input', ['no_atoms', 'radii', 'centres', 'adjacency_matrix'])

    return inputs(no_atoms=no_atoms, radii=radii, centres=centres, adjacency_matrix=adjacency_matrix)
