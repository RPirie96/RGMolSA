"""
script to clean up data set before computing descriptors

Functions:
- macrocycle_filter: removes macrocyclic molecules
- salt_filter: removes salts from molecules
- size_filter: removes mols too big or small
- drug_like_filter: removes non-drug like molecules
- ro5_filter: removes mols non-lipinski compliant
- pains_filter: removes molecules containing PAINS substructures
- filter_dataset: function to run above filters
"""

from collections import namedtuple
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def macrocycle_filter(mols, ids=None):

    """
    Function to remove molecules containing a macrocycle from data
    Macrocycles are defined here as any ring-like structure containing more than 12 atoms

    @param mols:
    @param ids:
    @return: named tuple of mols + ids with macrocyclic compounds removed

    Reference:
    Yudin, A. K. Macrocycles: lessons from the distant past, recent developments,
    and future directions. Chem. Sci. 6, 30–49 (2015).

    Source:
    Macrocycles with SMARTS queries - https://www.rdkit.org/docs/Cookbook.html
    """

    macro = Chem.MolFromSmarts("[r{12-}]")  # SMARTS pattern with ring size > 12
    mols_new, ids_new = [], []

    for i, mol in enumerate(mols):
        if not mol.GetSubstructMatches(macro):  # if there are no substructure matches
            mols_new.append(mol)  # add non-macro to list
            if ids is not None:
                ids_new.append(ids[i])  # add molecule id to list

    filtered_data = namedtuple("filtered_data", ["mols_new", "ids_new"])

    return filtered_data(mols_new=mols_new, ids_new=ids_new)


def salt_filter(mols):

    """
    Function to remove salts using the SuperParent option from the rdMolStandardize module of RDKit.
    The SuperParent is the fragment, charge, isotope, stereo, and tautomer parent of the molecule.

    @param mols:
    @return: list of molecules with salts removed

    Source:
    rdMolStandardize module - https://www.rdkit.org/docs/source/rdkit.Chem.MolStandardize.html
    """

    parent_mols = []
    non_bond = "."

    for mol in mols:
        smile = Chem.MolToSmiles(mol)
        if non_bond in smile:  # "non-bond" = "." indicates salt
            parent_mols.append(Chem.MolStandardize.rdMolStandardize.SuperParent(mol))
        else:
            parent_mols.append(mol)  # if the molecule doesn't have 2 parts add to list

    return parent_mols


def size_filter(mols, ids=None):

    """
    Function to remove molecules larger or smaller than the scope of this project. Filters:
     - anything with <6 heavy atoms (smaller than benzene generally not useful)
     - MW > 750Da (too big, N.B. calculate exact MW inc. H atoms c.f. ChEMBL)

    @param mols:
    @param ids:
    @return: named tuple of mols + ids of appropriate size
    """

    mols_new, ids_new = [], []

    for i, mol in enumerate(mols):
        if mol.GetNumHeavyAtoms() >= 6 and Descriptors.ExactMolWt(mol) <= 750.0:
            mols_new.append(mol)  # add appropriate sized mols to list
            if ids is not None:
                ids_new.append(ids[i])  # add molecule id to list

    filtered_data = namedtuple("filtered_data", ["mols_new", "ids_new"])

    return filtered_data(mols_new=mols_new, ids_new=ids_new)


def drug_like_filter(mols, ids=None):

    """
    Function to filter out non-drug like molecules. Defined as:
    - atoms outwith H, C, N, O, F, P, S, Cl, Br, I, B
    - > 7 B
    - 0 C

    @param mols:
    @param ids:
    @return: named tuple of mols + ids with non-druglike mols removed

    Reference:
    Filtering Chemical Libraries -
    http://practicalcheminformatics.blogspot.com/2018/08/filtering-chemical-libraries.html
    """

    # define list of acceptable atoms
    atom_list = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B"]
    mols_new, ids_new = [], []

    for i, mol in enumerate(mols):
        count_b = 0
        count_c = 0
        inc_mol = True
        for atom in mol.GetAtoms():
            a_type = atom.GetSymbol()
            if a_type not in atom_list:
                inc_mol = False
            if a_type == "B":  # count Boron
                count_b += 1
            if a_type == "C":  # count Carbon
                count_c += 1
        if count_b >= 7:
            inc_mol = False  # exclude mols with large no. of Boron atoms
        if count_c == 0:
            inc_mol = False  # exclude mols with no Carbons
        if inc_mol:
            mols_new.append(mol)  # add non-small to list
            if ids is not None:
                ids_new.append(ids[i])  # add molecule id to list

    filtered_data = namedtuple("filtered_data", ["mols_new", "ids_new"])

    return filtered_data(mols_new=mols_new, ids_new=ids_new)


def ro5_filter(mols, ids=None):

    """
    Function to carry out Lipinski's "Rule of 5" filtering. Conditions:
    - MW <= 500
    - no. HBAs <= 10
    - no. HBDs <= 5
    - LogP <= 5
    Molecule needs to pass at least 3/4 to be Lipinski compliant.

    @param mols:
    @param ids:
    @return: named tuple of mols + ids with Lipinski fails removed

    Reference:
    Lipinski, C. A., Lombardo, F., Dominy, B. W. & Feeney, P. J.
    Experimental and computational approaches to estimate solubility and permeability
    in drug discovery and development  settings.
    Advanced Drug Delivery Reviews 23, 3–25 (1997).
    """

    mols_new, ids_new = [], []
    for i, mol in enumerate(mols):
        mol_hs = Chem.AddHs(mol)

        # Calculate rule of five chemical properties
        mw = Descriptors.ExactMolWt(mol_hs)
        hba = Descriptors.NumHAcceptors(mol_hs)
        hbd = Descriptors.NumHDonors(mol_hs)
        logp = Descriptors.MolLogP(mol_hs)

        # Rule of five conditions
        conditions = [mw <= 500, hba <= 10, hbd <= 5, logp <= 5]

        if conditions.count(True) >= 3:  # add ro5 compliant to list
            mols_new.append(mol)
            if ids is not None:
                ids_new.append(ids[i])  # add molecule id to list

    filtered_data = namedtuple("filtered_data", ["mols_new", "ids_new"])

    return filtered_data(mols_new=mols_new, ids_new=ids_new)


def pains_filter(mols, ids=None):
    """
    Function to carry out PAINs filtering of a compound set.
    @param mols:
    @param ids:
    @return: named tuple of mols + ids with PAINs fails removed

    Reference:
    Baell, J. B. & Holloway, G. A. New Substructure Filters for
    Removal of Pan Assay Interference Compounds (PAINS) from Screening
    Libraries and for Their Exclusion in Bioassays.
    J. Med. Chem. 53, 2719–2740 (2010).
    """

    # initialize pains filter
    params_pains = Chem.rdfiltercatalog.FilterCatalogParams()
    params_pains.AddCatalog(
        Chem.rdfiltercatalog.FilterCatalogParams.FilterCatalogs.PAINS
    )
    catalog = Chem.FilterCatalog.FilterCatalog(params_pains)

    mols_new, ids_new = [], []
    for i, mol in enumerate(mols):
        entry = catalog.GetFirstMatch(mol)  # get the first matching PAINS
        if entry is None:  # if no matching substructures keep the molecule
            mols_new.append(mol)
            if ids is not None:
                ids_new.append(ids[i])  # add molecule id to list

    filtered_data = namedtuple("filtered_data", ["mols_new", "ids_new"])

    return filtered_data(mols_new=mols_new, ids_new=ids_new)


def filter_dataset(mols, ids=None, filename=None, ro5=False, pains=False):

    """
    Helper function to run all filters for dataset. Defaults:
    - macrocycle filter
    - size filter
    - salt filter
    - drug like filter
    Optional:
    - Lipinski's Rule of 5 Filter
    - PAINS filter
    If a file name is supplied, the filtered data is written to a SDF for future use

    @param pains:
    @param ro5:
    @param mols:
    @param ids:
    @param filename:
    @return: named tuple of filtered molecules and corresponding ids (if supplied)
    """

    if ids:
        filter_mac = macrocycle_filter(mols, ids)
        filter_parent = salt_filter(filter_mac.mols_new)
        filter_small = size_filter(filter_parent, filter_mac.ids_new)
        filter_drug = drug_like_filter(filter_small.mols_new, filter_small.ids_new)
        if ro5 and pains:
            filter_ro5 = ro5_filter(filter_drug.mols_new, filter_drug.ids_new)
            filter_pains = pains_filter(filter_ro5.mols_new, filter_ro5.ids_new)
            filtered_final = filter_pains
        elif ro5:
            filter_ro5 = ro5_filter(filter_drug.mols_new, filter_drug.ids_new)
            filtered_final = filter_ro5
        elif pains:
            filter_pains = pains_filter(filter_drug.mols_new, filter_drug.ids_new)
            filtered_final = filter_pains
        else:
            filtered_final = filter_drug

    else:
        filter_mac = macrocycle_filter(mols)
        filter_parent = salt_filter(filter_mac.mols_new)
        filter_small = size_filter(filter_parent)
        filter_drug = drug_like_filter(filter_small.mols_new)
        if ro5 and pains:
            filter_ro5 = ro5_filter(filter_drug.mols_new)
            filter_pains = pains_filter(filter_ro5.mols_new)
            filtered_final = filter_pains
        elif ro5:
            filter_ro5 = ro5_filter(filter_drug.mols_new)
            filtered_final = filter_ro5
        elif pains:
            filter_pains = pains_filter(filter_drug.mols_new)
            filtered_final = filter_pains
        else:
            filtered_final = filter_drug

    if filename:  # if a filename is supplied, save the filtered data to a SDF
        data = pd.DataFrame()
        if ids:
            data["ID"] = filtered_final.ids_new
        data["ROMol"] = filtered_final.mols_new

        Chem.PandasTools.WriteSDF(data, filename, molColName="ROMol")  # save to sdf

    return filtered_final
