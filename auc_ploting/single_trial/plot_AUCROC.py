# Import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, roc_auc_score
import os

#============================================================
# Define data information
feature_list = ['mf1024', 'mf2048', 'mol2vec', 'md', 'ensemble']
feature_name = ['SVM-ECFP1024', 'SVM-ECFP2048', 'XGB-Mol2Vec', 'RF-RDKit MD', 'iANP-EC']    
learning_algorithm = ['SVM', 'SVM', 'XGB', 'RF', 'top4']

# AUCROC plot
fpr_list, tpr_list, auc_list = [], [], []
for i in range(len(feature_list)):
    df = pd.read_csv('../result/pred/top4/y_prob_test_{}_{}_seed0.csv'.format(learning_algorithm[i], feature_list[i]), encoding='utf-8')
    fpr, tpr, _ = roc_curve(df['true_class'],  df['predicted_prob'])
    auc = np.round(roc_auc_score(df['true_class'], df['predicted_prob']), 4)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    auc_list.append(auc)

fig = plt.figure(figsize=(8,8))
for i in range(0,5):
    plt.plot(fpr_list[i], tpr_list[i], label="{}, AUROC={:.4f}".format(feature_name[i], auc_list[i]))
    plt.plot([0,1], [0,1], color='red', linestyle='--')
    
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('AUROC curves of iANP-EC and its base classifiers', fontsize=15)
plt.legend(prop={'size':10}, loc='lower right')
plt.show()

output_path = "../results/image/"
if os.path.isdir("../results/image/"):
    fig.savefig("../results/image/AUROC_{}.pdf".format('top4'))
else:
    os.makedirs(path)
    fig.savefig('../result/image/AUROC_{}.pdf'.format('top4'))

