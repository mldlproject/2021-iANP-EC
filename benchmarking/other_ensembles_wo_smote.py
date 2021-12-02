# Import libraries
from pyswarm import pso
import pandas as pd
import numpy as np
from imblearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import os

#============================================================
# Remove Low-variance Features
from sklearn.feature_selection import VarianceThreshold
threshold = (.95 * (1 - .95))

# Normalize Features
from sklearn.preprocessing import MinMaxScaler

#--------------------------------------------------------------------#
train_md      = np.load("../data/featurised_data/rdkit_md_train.npy")
train_mf1024  = np.load("../data/featurised_data/fp1024_train.npy")
train_mf2048  = np.load("../data/featurised_data/fp2048_train.npy")
train_mol2vec = np.load("../data/featurised_data/mol2vec_train.npy")
train_label   = np.load("../data/featurised_data/label_train.npy")

val_md      = np.load("../data/featurised_data/rdkit_md_val.npy")
val_mf1024  = np.load("../data/featurised_data/fp1024_val.npy")
val_mf2048  = np.load("../data/featurised_data/fp2048_val.npy")
val_mol2vec = np.load("../data/featurised_data/mol2vec_val.npy")
val_label   = np.load("../data/featurised_data/label_val.npy")

#--------------------------------------------------------------------#
my_svm_mf1024 = make_pipeline(VarianceThreshold(),
                              SVC(C=0.001, gamma=0.1, probability=True))

my_svm_mf2048 = make_pipeline(VarianceThreshold(),
                              SVC(C=0.001, gamma=0.1, probability=True))

my_rf_md = make_pipeline(MinMaxScaler(),
                         VarianceThreshold(),
                         RandomForestClassifier(n_estimators=100,
                                                max_depth=9,
                                                max_features=0.4,
                                                min_samples_split=3))

my_xgb_mol2vec = make_pipeline(VarianceThreshold(),
                               XGBClassifier(random_state=42, 
                                             n_estimators=100,
                                             max_depth=9,
                                             colsample_bytree=0.2,
                                             learning_rate=0.01))  

#--------------------------------------------------------------------#
val_pred_svm_mf1024   = my_svm_mf1024.fit(train_mf1024, train_label).predict_proba(val_mf1024)[::,1]
val_pred_svm_mf2048   = my_svm_mf2048.fit(train_mf2048, train_label).predict_proba(val_mf2048)[::,1]
val_pred_rf_md        = my_rf_md.fit(train_md, train_label).predict_proba(val_md)[::,1]
val_pred_xgb_mol2vec  = my_xgb_mol2vec.fit(train_mol2vec, train_label).predict_proba(val_mol2vec)[::,1]

#--------------------------------------------------------------------#
weight = pd.read_csv("../result/weights/roc_auc_test_top4ensemble_seed0.csv")['Weight']
weight
val_pred_top4ensemble = (val_pred_svm_mf1024*weight[0] +
                          val_pred_svm_mf2048*weight[1] + 
                          val_pred_rf_md*weight[2] + 
                          val_pred_xgb_mol2vec*weight[3])
#--------------------------------------------------------------------#
# Average Ensemble models
val_pred_mean = []
for i in range(len(val_pred_svm_mf1024)):
    pred = (val_pred_svm_mf1024[i] + val_pred_svm_mf2048[i] + val_pred_rf_md[i] + val_pred_xgb_mol2vec[i])/4
    val_pred_mean.append(round(pred,2))
    
val_pred_mean = np.array(val_pred_mean)

#--------------------------------------------------------------------#
# Major vote Ensemble models
val_pred_vote = []
for i in range(len(val_pred_svm_mf1024)):
    pred = (round(val_pred_svm_mf1024[i]) + round(val_pred_svm_mf2048[i]) + round(val_pred_rf_md[i]) + round(val_pred_xgb_mol2vec[i]))/4
    val_pred_vote.append(pred)
    
val_pred_vote = np.array(val_pred_vote)

#--------------------------------------------------------------------#
val_roc_auc_svm_mf1024    = roc_auc_score(val_label, val_pred_svm_mf1024)
val_roc_auc_svm_mf2048    = roc_auc_score(val_label, val_pred_svm_mf2048)
val_roc_auc_rf_md         = roc_auc_score(val_label, val_pred_rf_md)
val_roc_auc_xgb_mol2vec   = roc_auc_score(val_label, val_pred_xgb_mol2vec)
val_roc_auc_mean_ensemble = roc_auc_score(val_label, val_pred_mean)
val_roc_auc_vote_ensemble = roc_auc_score(val_label, val_pred_vote)
val_roc_auc_top4ensemble  = roc_auc_score(val_label, val_pred_top4ensemble)
 
#--------------------------------------------------------------------#
val_roc_list = [val_roc_auc_svm_mf1024, val_roc_auc_svm_mf2048, val_roc_auc_rf_md, val_roc_auc_xgb_mol2vec, val_roc_auc_mean_ensemble, val_roc_auc_vote_ensemble, val_roc_auc_top4ensemble]
fea_list = ['svm_mf1024', 'svm_mf2048', 'rf_md2048', 'xgb_mol2vec', 'mean_ensemble', 'vote_ensemble', 'top4ensemble']
df = pd.DataFrame(zip(fea_list, val_roc_list), columns=['model', 'AUCROC'])

output_path = "../results/benchmarking"
if os.path.isdir(output_path) is False:
        os.makedirs(output_path)
        
df.to_csv("{}/roc_auc_val_benchmarking_seed0.csv".format(output_path), index=False)