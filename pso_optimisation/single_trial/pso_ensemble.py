import pandas as pd
import numpy as np
import random
import os
from pyswarm.pyswarm import pso
from utils import *

from imblearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
#============================================================
# Remove Low-variance Features
from sklearn.feature_selection import VarianceThreshold
threshold = (.95 * (1 - .95))

# Normalize Features
from sklearn.preprocessing import MinMaxScaler

#============================================================
# Load data
train_md      = np.load("../../data/featurised_data/rdkit_md_train.npy")
train_mf1024  = np.load("../../data/featurised_data/fp1024_train.npy")
train_mf2048  = np.load("../../data/featurised_data/fp2048_train.npy")
train_mol2vec = np.load("../../data/featurised_data/mol2vec_train.npy")
train_label   = np.load("../../data/featurised_data/label_train.npy")

val_md        = np.load("../../data/featurised_data/rdkit_md_val.npy")
val_mf1024    = np.load("../../data/featurised_data/fp1024_val.npy")
val_mf2048    = np.load("../../data/featurised_data/fp2048_val.npy")
val_mol2vec   = np.load("../../data/featurised_data/mol2vec_val.npy")
val_label     = np.load("../../data/featurised_data/label_val.npy")

test_md       = np.load("../../data/featurised_data/rdkit_md_test.npy")
test_mf1024   = np.load("../../data/featurised_data/fp1024_test.npy")
test_mf2048   = np.load("../../data/featurised_data/fp2048_test.npy")
test_mol2vec  = np.load("../../data/featurised_data/mol2vec_test.npy")
test_label    = np.load("../../data/featurised_data/label_test.npy")

#============================================================
# Define base classifiers
my_svm_mf1024 = make_pipeline(VarianceThreshold(),
                              SVC(C=0.001, gamma=0.1, probability=True))

my_svm_mf2048 = make_pipeline(VarianceThreshold(),
                              SVC(C=0.001, gamma=0.1, probability=True))

my_rf_md = make_pipeline(MinMaxScaler(),
                         VarianceThreshold(),
                         RandomForestClassifier(random_state=42,
                                                n_estimators=100,
                                                max_depth=9,
                                                max_features=0.4,
                                                min_samples_split=3))

my_xgb_mol2vec = make_pipeline(VarianceThreshold(),
                               XGBClassifier(random_state=42, 
                                             n_estimators=100,
                                             max_depth=9,
                                             colsample_bytree=0.2,
                                             learning_rate=0.01))  

#============================================================
# Validation
val_pred_svm_mf1024  = my_svm_mf1024.fit(train_mf1024, train_label).predict_proba(val_mf1024)[::,1]
val_pred_svm_mf2048  = my_svm_mf2048.fit(train_mf2048, train_label).predict_proba(val_mf2048)[::,1]
val_pred_rf_md       = my_rf_md.fit(train_md, train_label).predict_proba(val_md)[::,1]
val_pred_xgb_mol2vec = my_xgb_mol2vec.fit(train_mol2vec, train_label).predict_proba(val_mol2vec)[::,1]

#============================================================
# Optimization
lb, ub = [0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5]

weight_list = []

seed_range = np.arange(0,10)

for s in seed_range:
    np.random.seed(s)
    random.seed(s)
    optx, fopt = pso(auc_optimasation, lb, ub, swarmsize=100, seed=s, maxiter=50)
    weight_list.append(extract_weight(optx))
    print("Round {}: Completed".format(s+1))

w1_list, w2_list, w3_list, w4_list = [], [], [], []
for weight in weight_list:
    w1_list.append(weight[0])
    w2_list.append(weight[1])
    w3_list.append(weight[2])
    w4_list.append(weight[3])

w1_avage_norm = np.round(np.mean(w1_list), 4)
w2_avage_norm = np.round(np.mean(w2_list), 4)
w3_avage_norm = np.round(np.mean(w3_list), 4)
w4_avage_norm = np.round(np.mean(w4_list), 4)
#============================================================
# Testing 
test_pred_svm_mf1024   = my_svm_mf1024.fit(np.concatenate([train_mf1024, val_mf1024]), np.concatenate([train_label, val_label])).predict_proba(test_mf1024)[::,1]
test_pred_svm_mf2048   = my_svm_mf2048.fit(np.concatenate([train_mf2048, val_mf2048]), np.concatenate([train_label, val_label])).predict_proba(test_mf2048)[::,1]
test_pred_rf_md        = my_rf_md.fit(np.concatenate([train_md, val_md]), np.concatenate([train_label, val_label])).predict_proba(test_md)[::,1]
test_pred_xgb_mol2vec  = my_xgb_mol2vec.fit(np.concatenate([train_mol2vec, val_mol2vec]), np.concatenate([train_label, val_label])).predict_proba(test_mol2vec)[::,1]
test_pred_top4ensemble = (test_pred_svm_mf1024*w1_avage_norm + test_pred_svm_mf2048*w2_avage_norm + test_pred_rf_md*w3_avage_norm + test_pred_xgb_mol2vec*w4_avage_norm)

test_roc_auc_svm_mf1024   = roc_auc_score(test_label, test_pred_svm_mf1024)
test_roc_auc_svm_mf2048   = roc_auc_score(test_label, test_pred_svm_mf2048)
test_roc_auc_rf_md        = roc_auc_score(test_label, test_pred_rf_md)
test_roc_auc_xgb_mol2vec  = roc_auc_score(test_label, test_pred_xgb_mol2vec)
test_roc_auc_top4ensemble = roc_auc_score(test_label, test_pred_top4ensemble)
 
#============================================================
# Export results
test_roc_list = [test_roc_auc_svm_mf1024, test_roc_auc_svm_mf2048, test_roc_auc_rf_md, test_roc_auc_xgb_mol2vec, test_roc_auc_top4ensemble]
weight_list   = [w1_avage_norm, w2_avage_norm, w3_avage_norm, w4_avage_norm, 1]
fea_list = ['svm_mf1024', 'svm_mf2048', 'rf_md2048', 'xgb_mol2vec', 'top4ensemble']

weight_path = '../../results/weights'
if os.path.isdir(weight_path) is False:
        os.makedirs(weight_path)

pred_path = '../../results/pred/top4'
if os.path.isdir(pred_path) is False:
        os.makedirs(pred_path)
         
pd.DataFrame(zip(test_roc_list,weight_list), index=fea_list, columns=['ROC-AUC', 'Weight']).to_csv("{}/roc_auc_test_top4ensemble_seed0.csv".format(weight_path),  index=None)
pd.DataFrame(zip(test_pred_svm_mf1024, test_label),   columns=["predicted_prob", "true_class"]).to_csv("{}/y_prob_test_SVM_mf1024_seed0.csv".format(pred_path),   index=None)
pd.DataFrame(zip(test_pred_svm_mf2048, test_label),   columns=["predicted_prob", "true_class"]).to_csv("{}/y_prob_test_SVM_mf2048_seed0.csv".format(pred_path),   index=None)
pd.DataFrame(zip(test_pred_rf_md, test_label),        columns=["predicted_prob", "true_class"]).to_csv("{}/y_prob_test_RF_md_seed0.csv".format(pred_path),        index=None)
pd.DataFrame(zip(test_pred_xgb_mol2vec, test_label),  columns=["predicted_prob", "true_class"]).to_csv("{}/y_prob_test_XGB_mol2vec_seed0.csv".format(pred_path),  index=None)
pd.DataFrame(zip(test_pred_top4ensemble, test_label), columns=["predicted_prob", "true_class"]).to_csv("{}/y_prob_test_top4ensemble_seed0.csv".format(pred_path), index=None)
