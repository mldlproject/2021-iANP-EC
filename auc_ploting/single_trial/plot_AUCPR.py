# Import libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.metrics import precision_recall_curve, average_precision_score
import os

#============================================================
# Define data information
feature_list = ['mf1024', 'mf2048', 'mol2vec', 'md', 'ensemble']
feature_name = ['SVM-ECFP1024', 'SVM-ECFP2048', 'XGB-Mol2Vec', 'RF-RDKit MD', 'iANP-EC']    
learning_algorithm = ['SVM', 'SVM', 'XGB', 'RF', 'top4']

# AUCROC plot
precision_list, recall_list, auc_list = [], [], []
for i in range(len(feature_list)):
    df = pd.read_csv('../result/pred/top4/y_prob_test_{}_{}_seed0.csv'.format(learning_algorithm[i], feature_list[i]), encoding='utf-8')
    precision, recall, _ = precision_recall_curve(df['true_class'],  df['predicted_prob'])
    auc = np.round(average_precision_score(df['true_class'], df['predicted_prob']), 4)
    precision_list.append(precision)
    recall_list.append(recall)
    auc_list.append(auc)

fig = plt.figure(figsize=(8,8))
for i in range(0,5):
    plt.plot(recall_list[i], precision_list[i], label="{}, AUPRC={:.4f}".format(feature_name[i], auc_list[i]))
    #plt.plot([1,0], [0,1], color='blue', linestyle='--')
    
plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)
plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)
plt.title('AUPRC curves of iANP-EC and its base classifiers', fontsize=15)
plt.legend(prop={'size':10}, loc='lower left')
plt.show()

output_path = "../results/image/"
if os.path.isdir(output_path):
    fig.savefig("../result/image/AUPRC_{}.pdf".format('top4'))
else:
    os.makedirs(output_path)
    fig.savefig("../result/image/AUPRC_{}.pdf".format('top4'))

