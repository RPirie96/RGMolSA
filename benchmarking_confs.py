#!/usr/bin/env
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
f = open("cxcr4_confs_log.txt", "a")

start_time = time.time()

#import actives
actives = PandasTools.LoadSDF("/home/b9046648/RGMolSA/cxcr4_actives_confs.sdf")

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


