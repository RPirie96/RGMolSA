"""
Script to compute conformers

Functions:
- embed_3d: generate single conformer from rdkit molecule structure
- embed_multi_3d: generate multiple conformers from rdkit molecule structure
"""

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign


def embed_3d(mol):

    """
    Function to generate a single 3D conformer of a molecule

    @param mol:
    @return: 3D structure for molecule
    """

    # set parameters for embedding
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True  # use random coordinates to help with issue of failed embeddings
    params.randomSeed = 0xf00d  # random seed for reproducibility
    params.useSmallRingTorsions = True  # includes recent improvements for small rings
    params.maxAttempts = 1000  # ignore smoothing failures, should be sufficient in most cases

    mol = Chem.AddHs(mol)  # add explicit Hs to obtain better conformations
    AllChem.EmbedMolecule(mol, params)  # get 3D coordinates
    mol = Chem.RemoveHs(mol)  # remove Hs

    return mol


def embed_multi_3d(mols, ids, filename, no_confs=None, energy_sorted=False):

    """
    Function to write multiple conformers of molecule to SDF. Number of conformers depends
    on no. rotatable bonds.

    @param mols:
    @param ids:
    @param filename:
    @param no_confs:
    @param energy_sorted:

    Reference:
    Ebejer, J.-P., Morris, G. M. & Deane, C. M.
    Freely Available Conformer Generation Methods: How Good Are They?
    J. Chem. Inf. Model. 52, 1146–1158 (2012).
    """

    # set up writer
    w = Chem.SDWriter(filename)

    # set parameters for embedding
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True  # use random coordinates to help with issue of failed embeddings
    params.randomSeed = 0xf00d  # random seed for reproducibility
    params.useSmallRingTorsions = True  # includes recent improvements for small rings
    params.maxAttempts = 1000  # ignore smoothing failures, should be sufficient in most cases
    params.pruneRmsThresh = 0.1  # ignore any confs that are too similar

    for i, mol in enumerate(mols):
        mol_id = ids[i]

        # add Hs to get better conformers
        mol = Chem.AddHs(mol)

        # get no. confs from no. rotatable bonds
        if no_confs is None:
            nb_rot_bonds = AllChem.CalcNumRotatableBonds(mol)
            if nb_rot_bonds <= 7:
                no_confs = 50  # low no conformers for low flexibility
            elif nb_rot_bonds <= 12:
                no_confs = 200  # more for higher flexibility
            else:
                no_confs = 300  # > 12 rotatable bonds need large no confs

        # get conformers
        cids = AllChem.EmbedMultipleConfs(mol, no_confs, params)

        # energy optimise conformers
        mol_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant='MMFF94s')

        if energy_sorted:
            # sort
            res = []
            for cid in cids:
                force_field = AllChem.MMFFGetMoleculeForceField(mol, mol_props, confId=cid)
                energy = force_field.CalcEnergy()
                res.append((cid, energy))
            sorted_res = sorted(res, key=lambda x: x[1])
            rdMolAlign.AlignMolConformers(mol)

            # remove Hs
            mol = Chem.RemoveHs(mol)

            for cid, energy in sorted_res:
                mol.SetProp('ID', str(mol_id))
                mol.SetProp('CID', str(cid))
                mol.SetProp('Energy', str(energy))
                w.write(mol, confId=cid)

        else:
            # remove Hs
            mol = Chem.RemoveHs(mol)
            for cid in cids:
                mol.SetProp('ID', str(mol_id))
                mol.SetProp('CID', str(cid))
                w.write(mol, confId=cid)

    w.close()
