# Import libraries
from feature_extractor.feature_extract import *
import pandas as pd
import numpy as np
import os

#============================================================
for i in range(10):
    datasets = ['train', 'val', 'test']
    output_path = "../data/ten_seeds/seed_{}/featurised_data/".format(i)
    if os.path.isdir(output_path) is False:
        os.makedirs(output_path)
    for dataset in datasets:
        data  = pd.read_csv("../data/ten_seeds/seed_{}/refined_data/data_{}.csv".format(i, dataset))['Smiles'].to_list()
        label = pd.read_csv("../data/ten_seeds/seed_{}/refined_data/data_{}.csv".format(i, dataset))['label']
        np.save("../data/ten_seeds/seed_{}/featurised_data/mol2vec_{}.npy".format(i, dataset), Mol2Vec_extract(data))
        np.save("../data/ten_seeds/seed_{}/featurised_data/fp1024_{}.npy".format(i, dataset), fp_extract(data, 1024))
        np.save("../data/ten_seeds/seed_{}/featurised_data/fp2048_{}.npy".format(i, dataset), fp_extract(data, 2048))
        np.save("../data/ten_seeds/seed_{}/featurised_data/label_{}.npy".format(i, dataset), label)
        #--------------------------------
        rdkit_data = pd.read_csv("../data/ten_seeds/seed_{}/rdkit_md/rdkit_md_{}.csv".format(i, dataset))
        np.save("../data/ten_seeds/seed_{}/featurised_data/rdkit_md_{}.npy".format(i, dataset), rdkit_data)
    
