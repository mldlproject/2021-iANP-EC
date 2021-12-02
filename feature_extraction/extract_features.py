# Import libraries
from feature_extractor.feature_extract import *
import pandas as pd
import numpy as np
import os

#============================================================
output_path = "../data/featurised_data/"
if os.path.isdir(output_path) is False:
    os.makedirs(output_path)
        
datasets = ['train', 'val', 'test']
for dataset in datasets:
    data = pd.read_csv("../data/refined_data/split_by_kmean/data_{}.csv".format(dataset))['Smiles'].to_list()
    label = pd.read_csv("../data/refined_data/split_by_kmean/data_{}.csv".format(dataset))['label']
    np.save("../data/featurised_data/mol2vec_{}.npy".format(dataset), Mol2Vec_extract(data))
    np.save("../data/featurised_data/fp1024_{}.npy".format(dataset), fp_extract(data, 1024))
    np.save("../data/featurised_data/fp2048_{}.npy".format(dataset), fp_extract(data, 2048))
    np.save("../data/featurised_data/label_{}.npy".format(dataset), label)
    #--------------------------------
    rdkit_data = pd.read_csv("../data/rdkit_md/rdkit_md_{}.csv".format(dataset))
    np.save("../data/featurised_data/rdkit_md_{}.npy".format(dataset), rdkit_data)
    
