# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

#============================================================
rdkit_md_train = pd.read_csv("../data/rdkit_md/rdkit_md_train.csv")
rdkit_md_val   = pd.read_csv("../data/rdkit_md/rdkit_md_val.csv")
rdkit_md_test  = pd.read_csv("../data/rdkit_md/rdkit_md_test.csv")
rdkit_md_data  = pd.concat([rdkit_md_train, rdkit_md_val, rdkit_md_test], axis=0).reset_index(drop=True)

label_train = np.load("../data/featurised_data/label_train.npy").tolist()
label_val   = np.load("../data/featurised_data/label_val.npy").tolist()
label_test  = np.load("../data/featurised_data/label_test.npy").tolist()

labels = []
for l in (label_train + label_val + label_test):
    if l == 0:
        labels.append('Non-anticarcinogen')
    else:
        labels.append('Anticarcinogen')
        
labels      = pd.DataFrame(labels, columns=['labels'])
labels_c    = pd.DataFrame(label_train + label_val + label_test, columns=['labels_c'])

dataset = ['Training']*len(label_train) + ['PSO-validation']*len(label_val) + ['Test']*len(label_test)
dataset = pd.DataFrame(dataset, columns=['Dataset'])

dataset_color = [1]*len(label_train) + [2]*len(label_val) + [3]*len(label_test)
dataset_color = pd.DataFrame(dataset_color, columns=['Dataset_color'])

# Standardizing the features
x = StandardScaler().fit_transform(rdkit_md_data)

# PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
principal_df = pd.DataFrame(data = principal_components
             , columns = ['PC1', 'PC2'])
final_df = pd.concat([principal_df, dataset, dataset_color, labels_c, labels], axis = 1)

# PCA visualization for datasets
plt.figure(figsize=(6,6))
plot1 = plt.scatter(final_df.iloc[:,0], final_df.iloc[:,1], c=final_df.iloc[:,3], cmap = sns.color_palette("flare", as_cmap=True), alpha=0.7)
plt.legend(handles=plot1.legend_elements()[0], labels=list(set(final_df['Dataset'])))
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Chemical Diversity")
plt.show()

# PCA visualization for samples
plt.figure(figsize=(6,6))
plot2 = plt.scatter(final_df.iloc[:,0], final_df.iloc[:,1], c=final_df.iloc[:,4], cmap = sns.color_palette("Spectral", as_cmap=True), alpha=0.7)
plt.legend(handles=plot2.legend_elements()[0], labels=list(set(final_df['labels'])))
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Chemical Diversity")
plt.show()
