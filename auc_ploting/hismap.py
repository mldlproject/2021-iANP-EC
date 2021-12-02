# Import library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#============================================================
svm_mf1024  = pd.read_csv("../result/ten_seeds/combined_pred_svm_mf1024_ten_seeds.csv")['trial_1']
svm_mf2048  = pd.read_csv("../result/ten_seeds/combined_pred_svm_mf2048_ten_seeds.csv")['trial_1']
rf_md       = pd.read_csv("../result/ten_seeds/combined_pred_rf_md_ten_seeds.csv")['trial_1']
xgb_mol2vec = pd.read_csv("../result/ten_seeds/combined_pred_xgb_mol2vec_ten_seeds.csv")['trial_1']
meta_top4   = pd.read_csv("../result/ten_seeds/combined_pred_meta_top4_ten_seeds.csv")['trial_1']

df = pd.concat([svm_mf1024, svm_mf2048, rf_md, xgb_mol2vec, meta_top4], axis=1)
df.columns = ['SVM-ECFP1024', 'SVM-ECFP2048', 'RF-RDKit MD', 'XGB-Mol2Vec', 'iANP-EC']

sns.heatmap(df.corr())
plt.show()