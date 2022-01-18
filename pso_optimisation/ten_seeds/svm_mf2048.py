# Import libraries
import pandas as pd
import numpy as np
from imblearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
import statistics as sts
import math

#============================================================
# Remove Low-variance Features
from sklearn.feature_selection import VarianceThreshold
threshold = (.95 * (1 - .95))

#============================================================
result_list=[]
for i in range(10):
    train_mf2048  = np.load("../../data/ten_seeds/seed_{}/featurised_data/fp2048_train.npy".format(i))
    train_label   = np.load("../../data/ten_seeds/seed_{}/featurised_data/label_train.npy".format(i))
    val_mf2048    = np.load("../../data/ten_seeds/seed_{}/featurised_data/fp2048_val.npy".format(i))
    val_label     = np.load("../../data/ten_seeds/seed_{}/featurised_data/label_val.npy".format(i))
    test_mf2048   = np.load("../../data/ten_seeds/seed_{}/featurised_data/fp2048_test.npy".format(i))
    test_label    = np.load("../../data/ten_seeds/seed_{}/featurised_data/label_test.npy".format(i))
    metric_df     = pd.read_csv("../../models_tuning/ten_seeds/mf_model_tuning/ten_seeds_results/seed_{}/result/svm/SVM_mf_2048_seed{}.csv".format(i, i)).iloc[11:,1].reset_index(drop=True)
    my_svm_mf2048 = make_pipeline(VarianceThreshold(), SVC(C=metric_df[0], gamma=metric_df[1], probability=True)) 
    test_pred_mf2048     = my_svm_mf2048.fit(np.concatenate([train_mf2048, val_mf2048]), np.concatenate([train_label, val_label])).predict_proba(test_mf2048)[::,1]
    test_roc_auc_mf2048  = roc_auc_score(test_label, test_pred_mf2048)
    result_list.append(test_roc_auc_mf2048)

mean = round(sts.mean(result_list),4)
sd   = round(sts.stdev(result_list), 4)
ci95 = ( round((mean - 1.96*sd/math.sqrt(10)), 4), round((mean + 1.96*sd/math.sqrt(10)),4))

