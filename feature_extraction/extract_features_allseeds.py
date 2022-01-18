#Import libraries
import pandas as pd
import numpy as np
import os
from featurizer.Mol2vec.getMol2vec import * 
from featurizer.rdkitMD.getRdkitMD import * 
from featurizer.MorganFP.getMorganFP import * 

#Extract features
for i in range(10):
    datasets = ['train', 'val', 'test']
    for dataset in datasets:
        data  = pd.read_csv("../data/allseeds_kmean/seed_{}/data_{}.csv".format(i, dataset))['SMILES'].to_list()
        label = pd.read_csv("../data/allseeds_kmean/seed_{}/data_{}.csv".format(i, dataset))['class']
        #--------------------------------
        PATH = "../data/allseeds/seed_{}/featurised_data".format(i)
        if os.path.isdir(PATH) == False:
            os.makedirs(PATH)
        #--------------------------------
        rdkit_md = np.array(extract_rdkitMD(data).iloc[::,1:])
        mf1024   = np.array(extract_MorganFP(data, bit_type=1024).iloc[::,1:])
        mf2048   = np.array(extract_MorganFP(data, bit_type=1024).iloc[::,1:])
        mol2vec  = extract_Mol2Vec(data)
        #--------------------------------
        np.save('../data/allseeds/seed_{}/featurised_data/rdkit_md_{}.npy'.format(i, dataset), rdkit_md)
        np.save("../data/allseeds/seed_{}/featurised_data/fp1024_{}.npy".format(i, dataset), mf1024)
        np.save("../data/allseeds/seed_{}/featurised_data/fp2048_{}.npy".format(i, dataset), mf2048)
        np.save("../data/allseeds/seed_{}/featurised_data/mol2vec_{}.npy".format(i, dataset), mol2vec)
        np.save("../data/allseeds/seed_{}/featurised_data/label_{}.npy".format(i, dataset), label)