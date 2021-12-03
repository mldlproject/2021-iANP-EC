from utils import printPerformance
import pandas as pd
import numpy as np

baseline_clf = ['SVM_mf1024', 'SVM_mf2048', 'RF_md', 'XGB_mol2vec', 'top4ensemble']
metric = ['AUC-ROC', 'AUC-PR', 'ACC', 'BA', 'SN/RE', 'SP', 'PR', 'MCC', 'F1', 'CK']
empydf = pd.DataFrame(metric)

for clf in baseline_clf:
    probs  = np.array(pd.read_csv("../result/pred/top4/y_prob_test_{}_seed0.csv".format(clf))["predicted_prob"])
    labels = np.load("../data/featurised_data/label_test.npy")
    result = pd.DataFrame(list(printPerformance(labels, probs, decimal=4)))
    empydf = pd.concat([empydf, result], axis=1)
    
empydf.columns = ['metrics'] + baseline_clf
empydf.to_csv("../result/combined_result.csv", index=False)   

