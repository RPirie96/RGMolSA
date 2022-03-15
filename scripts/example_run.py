#!/usr/bin/python
"""
Example script for running full RGMolSA method including conformer generation and data filtering
"""

import sys
import pandas as pd
from rdkit.Chem import PandasTools

from data_filters import filter_dataset
from conf_gen import embed_multi_3d
from get_descriptor import get_descriptor
from utils import get_score


def get_cl_inputs():
    """
    helper function to get the filenames for the initial set and to write confs too from command line + csv filename
    @return:
    """
    mol_set = sys.argv[1]
    conf_filename = sys.argv[2]
    scores_name = sys.argv[3]

    return mol_set, conf_filename, scores_name


def run_descriptors():

    # get the dataset and conformer filenames
    mol_set, conf_filename, scores_name = get_cl_inputs()

    # load dataset
    data = PandasTools.LoadSDF(mol_set)
    mols = list(data['ROMol'])
    ids = list(data['ID'])

    # filter the data
    filtered = filter_dataset(mols, ids)  # add filename to save, ro5/pains = True for additional filters
    mols = filtered.mols_new
    ids = filtered.ids_new

    # generate conformers
    embed_multi_3d(mols, ids, conf_filename, no_confs=None, energy_sorted=False)

    # load conformers
    data = PandasTools.LoadSDF(conf_filename)
    mols = list(data['ROMol'])
    ids = list(data['ID'])

    # get descriptor for each molecule
    descriptors = [get_descriptor(mol) for mol in mols]

    # get scores, treating first molecule as query
    query = descriptors[0]
    qid = ids[0]

    scores = []
    for i, des in enumerate(descriptors):
        scores.append(get_score(query, des, qid, ids[i]))

    # create dataframe of ID vs score, sort with highest first and save to csv
    scores_df = pd.Dataframe(list(zip(ids, scores)), columns=['ID', 'Score'])
    scores_df = scores_df[scores_df['Score'] != 'self']
    scores_df = scores_df.sort_values('scores', ascending=False)
    scores_df = scores_df.reset_index(drop=True)

    scores_df.to_csv(scores_name)


run_descriptors()
