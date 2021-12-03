# Import libraries
import pandas as pd
import numpy as np
from imblearn.pipeline import make_pipeline
from xgboost import XGBClassifier
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
    train_mol2vec  = np.load("../../data/ten_seeds/seed_{}/featurised_data/mol2vec_train.npy".format(i))
    train_label    = np.load("../../data/ten_seeds/seed_{}/featurised_data/label_train.npy".format(i))
    val_mol2vec    = np.load("../../data/ten_seeds/seed_{}/featurised_data/mol2vec_val.npy".format(i))
    val_label      = np.load("../../data/ten_seeds/seed_{}/featurised_data/label_val.npy".format(i))
    test_mol2vec   = np.load("../../data/ten_seeds/seed_{}/featurised_data/mol2vec_test.npy".format(i))
    test_label     = np.load("../../data/ten_seeds/seed_{}/featurised_data/label_test.npy".format(i))
    metric_df      = pd.read_csv("../../models_tuning/ten_seeds/mol2vec_model_tuning/ten_seeds_results/seed_{}/result/xgb/XGB_mol2vec_seed{}.csv".format(i, i)).iloc[11:,1].reset_index(drop=True)
    my_xgb_mol2vec = make_pipeline(VarianceThreshold(),
                                   XGBClassifier(random_state=42, 
                                                 n_estimators=100,
                                                 max_depth=int(metric_df[0]),
                                                 colsample_bytree=metric_df[1],
                                                 learning_rate=metric_df[2]))  
    test_pred_mol2vec     = my_xgb_mol2vec.fit(np.concatenate([train_mol2vec, val_mol2vec]), np.concatenate([train_label, val_label])).predict_proba(test_mol2vec)[::,1]
    test_roc_auc_mol2vec  = roc_auc_score(test_label, test_pred_mol2vec)
    result_list.append(test_roc_auc_mol2vec)


mean = round(sts.mean(result_list),4)
sd   = round(sts.stdev(result_list), 4)
ci95 = ( round((mean - 1.96*sd/math.sqrt(10)), 4), round((mean + 1.96*sd/math.sqrt(10)),4))
