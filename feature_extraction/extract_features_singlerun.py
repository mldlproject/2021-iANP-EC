#Import libraries
import pandas as pd
import numpy as np
import os
from featurizer.Mol2vec.getMol2vec import * 
from featurizer.rdkitMD.getRdkitMD import * 
from featurizer.MorganFP.getMorganFP import * 

#Extract features
datasets = ['train', 'val', 'test']
for dataset in datasets:
    data  = pd.read_csv("../data/singlerun_kmean/data_{}.csv".format(dataset))['Smiles'].to_list()
    label = pd.read_csv("../data/singlerun_kmean/data_{}.csv".format(dataset))['label']
    #--------------------------------
    PATH = "../data/singlerun/featurised_data"
    if os.path.isdir(PATH) == False:
        os.makedirs(PATH)
    #--------------------------------
    rdkit_md = np.array(extract_rdkitMD(data).iloc[::,1:])
    mf1024   = np.array(extract_MorganFP(data, bit_type=1024).iloc[::,1:])
    mf2048   = np.array(extract_MorganFP(data, bit_type=2048).iloc[::,1:])
    mol2vec  = extract_Mol2Vec(data)
    #--------------------------------
    np.save('../data/singlerun/featurised_data/rdkit_md_{}.npy'.format(dataset), rdkit_md)
    np.save('../data/singlerun/featurised_data/fp1024_{}.npy'.format(dataset), mf1024)
    np.save('../data/singlerun/featurised_data/fp2048_{}.npy'.format(dataset),mf2048)
    np.save('../data/singlerun/featurised_data/mol2vec_{}.npy'.format(dataset), mol2vec)
    np.save('../data/singlerun/featurised_data/label_{}.npy'.format(dataset), label)