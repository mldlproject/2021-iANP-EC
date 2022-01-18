# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

#============================================================
rdkit_md_anp    = pd.read_csv("../data/rdkit_md/all_smiles.csv").iloc[:,:-1]
rdkit_md_anp['label'] = "iANP-EC"

rdkit_md_npass  = pd.read_csv("../data/rdkit_md/npass_rdkitmd.csv")
rdkit_md_npass['label'] = "Natural Product Space"

merged_anp_npass = pd.concat([rdkit_md_anp, rdkit_md_npass], axis=0)
merged_anp_npass = merged_anp_npass.reset_index(drop=True)

merged_anp_npass = merged_anp_npass.drop_duplicates(subset=['SMILES'], keep='first')
      
source = merged_anp_npass['label']
source_color = pd.DataFrame([0]*len(merged_anp_npass[merged_anp_npass['label']=='iANP-EC']) + [1]*len(merged_anp_npass[merged_anp_npass['label']=='Natural Product Space']))
source_color.columns = ['source_color']

# Standardizing the features
x = StandardScaler().fit_transform(merged_anp_npass.iloc[:,1:-1])

notnan = []
nanIdx = []
notnanIdx = []

for i in range(len(x)):
    if np.any(np.isnan(x[i])) == False:
        notnan.append(x[i])
        notnanIdx.append(i)
    else:
        nanIdx.append(i)

x_notnan = pd.DataFrame(x).iloc[np.array(notnanIdx),:].reset_index(drop=True)
x_notnan = np.array(x_notnan)

# PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x_notnan)
principal_df = pd.DataFrame(data = principal_components
             , columns = ['PC1', 'PC2'])
final_df = pd.concat([principal_df, source.iloc[np.array(notnanIdx)].reset_index(drop=True), source_color.iloc[np.array(notnanIdx)].reset_index(drop=True)], axis = 1)

# PCA visualization for datasets
plt.figure(figsize=(10,10))
plt.scatter(final_df.iloc[997:,0], final_df.iloc[997:,1], label="Natural Product Space (Source: NPASS)", alpha=0.5, s=15, color='#6DB6FF')
plt.scatter(final_df.iloc[:997,0], final_df.iloc[:997,1], label="iANP-EC (All chemical data)", alpha=0.5, s=15, color='#094792')
plt.legend(prop={'size':10}, loc='upper left')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Chemical Diversity")
plt.show()
