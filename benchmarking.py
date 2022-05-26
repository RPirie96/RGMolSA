#!/usr/bin/env
import pandas as pd
import time
import pickle

from rdkit.Chem import PandasTools

import sys
sys.path.append('/home/b9046648/RGMolSA/scripts')

from get_descriptor import get_descriptor
from data_filters import macrocycle_filter, size_filter, neutralize_atoms
from utils import get_score

# text file to log time and excluded mols
f = open("xiap_log.txt", "a")

start_time = time.time()

# import active data from SDF
actives = PandasTools.LoadSDF('/home/b9046648/dud-e/other/xiap/xiap_actives.sdf')
actives = actives.drop_duplicates('ID', keep = 'first')
actives = actives.reset_index(drop=True)

ids = list(actives['ID'])
mols = list(actives['ROMol'])
f.write("Number of Actives pre-filtering {0} \n".format(len(mols)))
f.write("-----------------------------\n")

# remove macrocycles and molecules that are too big/small
mac_filtered = macrocycle_filter(mols, ids)
size_filtered = size_filter(mac_filtered.mols_new, mac_filtered.ids_new)

active_mols = size_filtered.mols_new

# neutralise molecules
for mol in active_mols:
    neutralize_atoms(mol)

active_ids = size_filtered.ids_new

# log updated no actives + list of IDs excluded
f.write("Number of Actives post filtering {0} \n".format(len(active_mols)))
f.write("-----------------------------\n")

filtered_ids = [i for i in ids if i not in active_ids]

f.write("Active molecules excluded by size/macrocycle filters: {0}\n".format(filtered_ids))
f.write("-----------------------------\n")

# get descriptors
active_descriptors = [get_descriptor(mol) for mol in active_mols]

# remove mols that raise error + log ids
error_idx = []
for i in range(len(active_descriptors)):
    if active_descriptors[i] == "TypeError":
        error_idx.append(i)
    if active_descriptors[i] == "ArithmeticError":
        error_idx.append(i)
    if active_descriptors[i] == "LinAlgError":
        error_idx.append(i)
    if active_descriptors[i] == "ValueError":
        error_idx.append(i)

error_ids = [active_ids[i] for i in error_idx]
if len(error_ids) != 0:
    f.write("Active molecules that raised a Arithmetic/Type/LinAlg/Value error: {0}\n".format(error_ids))
    f.write("-----------------------------\n")

active_descriptors = [active_descriptors[i] for i in range(len(active_descriptors)) if i not in error_idx]
active_ids = [active_ids[i] for i in range(len(active_ids)) if i not in error_idx]

# create new actives df of ids, descriptors and label True
actives = pd.DataFrame(list(zip(active_ids, active_descriptors)), columns=['ID', 'RGMolSA_Des'])
actives['type'] = True  # label for EF count

# import active data from SDF
decoys = PandasTools.LoadSDF('/home/b9046648/dud-e/other/xiap/xiap_inactives.sdf')
decoys = decoys.drop_duplicates('ID', keep = 'first')
decoys = decoys.reset_index(drop=True)

ids = list(decoys['ID'])
mols = list(decoys['ROMol'])
f.write("Number of Decoys pre-filtering {0}\n".format(len(mols)))
f.write("-----------------------------\n")

# remove macrocycles and molecules that are too big/small
mac_filtered = macrocycle_filter(mols, ids)
size_filtered = size_filter(mac_filtered.mols_new, mac_filtered.ids_new)

decoy_mols = size_filtered.mols_new

# neutralise mols
for mol in decoy_mols:
    neutralize_atoms(mol)

decoy_ids = size_filtered.ids_new

# log updated no actives + list of IDs excluded
f.write("Number of Decoys post filtering {0}\n".format(len(decoy_mols)))
f.write("-----------------------------\n")

filtered_ids = [i for i in ids if i not in decoy_ids]

f.write("Decoy molecules excluded by size/macrocycle filters: {0}\n".format(filtered_ids))
f.write("-----------------------------\n")

# get descriptors
decoy_descriptors = [get_descriptor(mol) for mol in decoy_mols]

# remove mols that raise error + log ids
error_idx = []
for i in range(len(decoy_descriptors)):
    if decoy_descriptors[i] == "TypeError":
        error_idx.append(i)
    if decoy_descriptors[i] == "ArithmeticError":
        error_idx.append(i)
    if decoy_descriptors[i] == "LinAlgError":
        error_idx.append(i)
    if decoy_descriptors[i] == "ValueError":
        error_idx.append(i)

error_ids = [decoy_ids[i] for i in error_idx]
if len(error_ids) != 0:
    f.write("Decoy molecules that raised a Arithmetic/Type/LinAlg/Value error: {0}\n".format(error_ids))
    f.write("-----------------------------\n")

decoy_descriptors = [decoy_descriptors[i] for i in range(len(decoy_descriptors)) if i not in error_idx]
decoy_ids = [decoy_ids[i] for i in range(len(decoy_ids)) if i not in error_idx]

# create new actives df of ids, descriptors and label True
decoys = pd.DataFrame(list(zip(decoy_ids, decoy_descriptors)), columns=['ID', 'RGMolSA_Des'])
decoys['type'] = False  # label for EF count

# calculate similarity scores
df_list = []

for i in range(len(active_descriptors)):
    query = active_descriptors[i]
    qid = active_ids[i]

    scores_active = [get_score(query, active_descriptors[i], qid, active_ids[i]) for i in range(len(active_descriptors))]
    scores_decoy = [get_score(query, decoy_descriptors[i], qid, decoy_ids[i]) for i in range(len(decoy_descriptors))]

    actives['scores'] = scores_active
    decoys['scores'] = scores_decoy

    # put results together
    frames = [actives, decoys]
    result = pd.concat(frames)
    result = result[result['scores'] != 'self']
    result = result.sort_values('scores', ascending=False)
    result = result.reset_index(drop=True)
    del result['RGMolSA_Des']

    df_list.append(result)

elapsed = time.time()
f.write("Run Time: {} \n".format(elapsed-start_time))
f.close()  # close log

# save dataframe
dictionary = dict(zip(active_ids, df_list))
with open('xiap_rgmolsa.pickle', 'wb') as handle:
    pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
