# Import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, roc_auc_score
import os

#============================================================
# Define data information
feature_list = ['mf1024', 'mf2048', 'md', 'mol2vec', 'top4']
feature_name = ['SVM-ECFP1024', 'SVM-ECFP2048', 'RF-RDKit MD', 'XGB-Mol2Vec', 'iANP-EC']    
learning_algorithm = ['SVM', 'SVM', 'RF', 'XGB', 'META']

# AUCROC plot
fpr_list, tpr_list, auc_list = [], [], []
for i in range(len(feature_list)):
    df = pd.read_csv("../../results/pso/allseeds/combined_pred_{}_{}_allseeds.csv".format(learning_algorithm[i].lower(), feature_list[i]), encoding='utf-8')
    true_class = pd.DataFrame(np.load("../../data/singlerun/featurised_data/label_test.npy"))
    fpr, tpr, _ = roc_curve(true_class,  df['trial_1'])
    auc = np.round(roc_auc_score(true_class, df['trial_1']), 4)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    auc_list.append(auc)

fig = plt.figure(figsize=(8,8))
for i in range(0,5):
    plt.plot(fpr_list[i], tpr_list[i], label="{}, ROC-AUC={:.4f}".format(feature_name[i], auc_list[i]))
    plt.plot([0,1], [0,1], color='red', linestyle='--')
    
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('ROC-AUC curves of iANP-EC and its base classifiers', fontsize=15)
plt.legend(prop={'size':10}, loc='lower right')
plt.show()

output_path = "../../results/image/"
if os.path.isdir(output_path):
    fig.savefig("../../results/image/AUROC_{}.pdf".format('top4'))
else:
    os.makedirs(output_path)
    fig.savefig("../../results/image/AUROC_{}.pdf".format('top4'))