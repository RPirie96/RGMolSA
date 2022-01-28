# script to get initial inputs
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from collections import namedtuple


def get_mol_info(mol):
    """
        Function which reads in ROMol structure, replaces rings with spheres and outputs no atoms, 3D centres,
        adjacency matrix and list of radii

        :param mol:
        :return adjacency_matrix, no_atoms, centres, radii:
    """

    # check for 3D coords, try embed if not
    if '3D' not in Chem.MolToMolBlock(mol).split("\n")[1]:  # check if coords are 3D, get conf if not
        ps = AllChem.ETKDGv3()
        ps.useRandomCoords = True  # use random coordinates to help with issue of failed embeddings
        ps.maxAttempts = 1000  # ignore smoothing failures, should be sufficient in most cases

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, ps)
        mol = Chem.RemoveHs(mol)

    # if embed still fails, raise TypeError
    if '3D' not in Chem.MolToMolBlock(mol).split("\n")[1]:  # check that the above embed has worked
        raise TypeError('RDKit cannot produce 3D coordinates for this molecule')

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
        indices = [a.GetIdx() for a in mol.GetAtoms()]  # list of all atom indices
        rings = [list(mol.GetRingInfo().AtomRings()[i]) for i in range(num_rings)]  # list of ring indices
        ring_pos = list(set(sorted(
            [index for sublist in rings for index in sublist])))  # this creates a sorted list of ring indexes
        non_ring = [x for x in indices if x not in ring_pos]
        updated_non_ring = [i + len(rings) for i in range(len(non_ring))]

        # update molecule information structures to account for rings

        # get a list of the atom centres contained in the ring
        ring_centres = [np.take(centres, ring, axis=0) for ring in rings]

        # get centroid of each ring
        centroids = np.array([[np.sum(ring[:, 0]) / len(ring[:, 0]), np.sum(ring[:, 1]) / len(ring[:, 1]),
                               np.sum(ring[:, 2]) / len(ring[:, 2])] for ring in ring_centres])

        # remove ring atoms and add in spheres at the beginning of each structure
        # choose beginning as easier to index using num_rings!

        no_atoms = no_atoms - len(ring_pos) + len(rings)  # update inputs to reflect ring --> sphere
        centres = np.vstack(
            (centroids, np.delete(centres, ring_pos, axis=0)))  # remove ring atom centres and add centroids
        radii = np.append(np.full(len(rings), 2.25), np.delete(radii, ring_pos))  # remove ring atom radii and add ring

        new_am = np.zeros((no_atoms, no_atoms), dtype=int)  # define empty matrix of new size
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
                    if adjacency_matrix[k][non_ring[l]] == 1 and new_am[i][updated_non_ring[l]] == 0:
                        new_am[i][updated_non_ring[l]] = 1
                        new_am[updated_non_ring[l]][i] = 1
        # intersection between non ring
        for i in range(len(non_ring)):
            for j in range(len(non_ring)):
                if adjacency_matrix[non_ring[i]][non_ring[j]] == 1 and new_am[updated_non_ring[i]][updated_non_ring[j]] == 0:
                    new_am[updated_non_ring[i]][updated_non_ring[j]] = 1
                    new_am[updated_non_ring[j]][updated_non_ring[i]] = 1
    else:
        new_am = adjacency_matrix
    inputs = namedtuple('input', ['no_atoms', 'radii', 'centres', 'adjacency_matrix'])

    return inputs(no_atoms=no_atoms, radii=radii, centres=centres, adjacency_matrix=new_am)