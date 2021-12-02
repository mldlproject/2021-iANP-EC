import numpy as np 
import pandas as pd 
from rdkit import Chem

active = pd.read_csv("data/AnticancerNP_active.csv")
inactive = pd.read_csv("data/AnticancerNP_inactive.csv")

filtered_active   = active.drop_duplicates(subset=['SMILES'], keep='first')
filtered_inactive = inactive.drop_duplicates(subset=['SMILES'], keep='first')

canonical_active = []
for smiles in filtered_active['SMILES'].tolist():
    canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    canonical_active.append(canonical_smiles)
    
df_canonical_active = pd.DataFrame(canonical_active, columns=["SMILES"])
filtered_canonical_active = df_canonical_active.drop_duplicates(subset=['SMILES'], keep='first')

canonical_inactive = []
for smiles in filtered_inactive['SMILES'].tolist():
    canonical_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    canonical_inactive.append(canonical_smiles)
    
df_canonical_inactive = pd.DataFrame(canonical_inactive, columns=["SMILES"])
filtered_canonical_inactive = df_canonical_inactive.drop_duplicates(subset=['SMILES'], keep='first')

filtered_canonical_active['Label']   = np.ones(len(filtered_canonical_active), dtype=int)
filtered_canonical_inactive['Label'] = np.zeros(len(filtered_canonical_inactive), dtype=int)

filtered_canonical_merged = pd.concat([filtered_canonical_active, filtered_canonical_inactive], axis=0, ignore_index=True)

refined_canonical_merged = filtered_canonical_merged.drop_duplicates(subset=['SMILES'], keep=False)

num_active   = len(refined_canonical_merged[refined_canonical_merged['Label']==1]) #372
num_inactive = len(refined_canonical_merged[refined_canonical_merged['Label']==0]) #639

pd.DataFrame.to_csv(refined_canonical_merged, "./data/refined_anticancerNP.csv", index=False)
