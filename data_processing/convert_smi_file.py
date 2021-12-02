# Import libraries
from utils import *
import pandas as pd

#============================================================
# Convert csv to smi files
for i in range(10):
    dataset = ['train', 'val', 'test']
    for d in dataset:
        smiles_list = pd.read_csv("../data/ten_seeds/seed_{}/refined_data/data_{}.csv".format(i, d), encoding='utf-8')['Smiles'].tolist()
        convert2smi(smiles_list, tag=d, output_path='../data/ten_seeds/seed_{}/refined_data'.format(i))
