# Import libraries
from pyswarm.pyswarm import pso
import pandas as pd
import numpy as np
import random
import os
import random
import statistics as sts
import math

from imblearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from utils import *

#============================================================
# Remove Low-variance Features
from sklearn.feature_selection import VarianceThreshold
threshold = (.95 * (1 - .95))

# Normalize Features
from sklearn.preprocessing import MinMaxScaler

#============================================================
normalized_weight_list = []

svm_mf1024_test_auc_list = []
svm_mf2048_test_auc_list = []
rf_md_test_auc_list = []
xgb_test_auc_list = []
meta_test_auc_list = []

svm_mf1024_pred_list = []
svm_mf2048_pred_list = []
rf_md_pred_list = []
xgb_mol2vec_pred_list = []
meta_pred_list = []

for i in range(30):
    # Load training data
    train_md      = np.load("../../data/allseeds/seed_{}/featurised_data/rdkit_md_train.npy".format(i))
    train_mf1024  = np.load("../../data/allseeds/seed_{}/featurised_data/fp1024_train.npy".format(i))
    train_mf2048  = np.load("../../data/allseeds/seed_{}/featurised_data/fp2048_train.npy".format(i))
    train_mol2vec = np.load("../../data/allseeds/seed_{}/featurised_data/mol2vec_train.npy".format(i))
    train_label   = np.load("../../data/allseeds/seed_{}/featurised_data/label_train.npy".format(i))
    #--------------------------------------------------------------------#
    # Load validation data
    val_md      = np.load("../../data/allseeds/seed_{}/featurised_data/rdkit_md_val.npy".format(i))
    val_mf1024  = np.load("../../data/allseeds/seed_{}/featurised_data/fp1024_val.npy".format(i))
    val_mf2048  = np.load("../../data/allseeds/seed_{}/featurised_data/fp2048_val.npy".format(i))
    val_mol2vec = np.load("../../data/allseeds/seed_{}/featurised_data/mol2vec_val.npy".format(i))
    val_label   = np.load("../../data/allseeds/seed_{}/featurised_data/label_val.npy".format(i))
    #--------------------------------------------------------------------#
    # Load test data
    test_md      = np.load("../../data/allseeds/seed_{}/featurised_data/rdkit_md_test.npy".format(i))
    test_mf1024  = np.load("../../data/allseeds/seed_{}/featurised_data/fp1024_test.npy".format(i))
    test_mf2048  = np.load("../../data/allseeds/seed_{}/featurised_data/fp2048_test.npy".format(i))
    test_mol2vec = np.load("../../data/allseeds/seed_{}/featurised_data/mol2vec_test.npy".format(i))
    test_label   = np.load("../../data/allseeds/seed_{}/featurised_data/label_test.npy".format(i))
    #--------------------------------------------------------------------#
    # Load best hyperparameters
    path = "C:/Users/nvtho/OneDrive/Desktop/DONE PROJECTS/2022-iANP-EC/model_tuning/allseeds"
    metric_svm_mf1024   = pd.read_csv("{}/mf_model_tuning/allseeds_results/seed_{}/result/svm/SVM_mf_1024_seed{}.csv".format(path, i, i)).iloc[11:,1].reset_index(drop=True)
    metric_svm_mf2048   = pd.read_csv("{}/mf_model_tuning/allseeds_results/seed_{}/result/svm/SVM_mf_2048_seed{}.csv".format(path, i, i)).iloc[11:,1].reset_index(drop=True)
    metric_rf_md        = pd.read_csv("{}/md_model_tuning/allseeds_results/seed_{}/result/rf/RF_md_seed{}.csv".format(path, i, i)).iloc[11:,1].reset_index(drop=True)
    metric_xgb_mol2vec  = pd.read_csv("{}/mol2vec_model_tuning/allseeds_results/seed_{}/result/xgb/XGB_mol2vec_seed{}.csv".format(path, i, i)).iloc[11:,1].reset_index(drop=True)
    #--------------------------------------------------------------------#
    # Create classifiers
    my_svm_mf1024 = make_pipeline(VarianceThreshold(),
                                SVC(C=metric_svm_mf1024[0], gamma=metric_svm_mf1024[1], probability=True)) 
    my_svm_mf2048 = make_pipeline(VarianceThreshold(),
                                SVC(C=metric_svm_mf1024[0], gamma=metric_svm_mf1024[1], probability=True))
    my_rf_md = make_pipeline(MinMaxScaler(),
                            VarianceThreshold(),
                            RandomForestClassifier(random_state=42,
                                                   n_estimators=100,
                                                   max_depth=int(metric_rf_md[0]),
                                                   max_features=metric_rf_md[1],
                                                   min_samples_split=int(metric_rf_md[2])))      
    my_xgb_mol2vec = make_pipeline(VarianceThreshold(),
                                   XGBClassifier(random_state=42, 
                                                 n_estimators=100,
                                                 max_depth=int(metric_xgb_mol2vec[0]),
                                                 colsample_bytree=metric_xgb_mol2vec[1],
                                                 learning_rate=metric_xgb_mol2vec[2]))
    #--------------------------------------------------------------------#
    # Get predicted probabilities of validation sets
    val_pred_svm_mf1024  = my_svm_mf1024.fit(train_mf1024, train_label).predict_proba(val_mf1024)[::,1]
    val_pred_svm_mf2048  = my_svm_mf2048.fit(train_mf2048, train_label).predict_proba(val_mf2048)[::,1]
    val_pred_rf_md       = my_rf_md.fit(train_md, train_label).predict_proba(val_md)[::,1]
    val_pred_xgb_mol2vec = my_xgb_mol2vec.fit(train_mol2vec, train_label).predict_proba(val_mol2vec)[::,1]
    #--------------------------------------------------------------------#
    val_preds = [val_pred_svm_mf1024, val_pred_svm_mf2048, val_pred_rf_md, val_pred_xgb_mol2vec]
    auc_optimasation = get_optimasation_function(val_preds, val_label)
    #--------------------------------------------------------------------#
    # Define parameters for optimization
    lb, ub = [0.1, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5]
    weight_list = []
    seed_range = np.arange(0,10)
    # Optimization loop
    for s in seed_range:
        np.random.seed(s)
        random.seed(s)
        optx, fopt = pso(auc_optimasation, lb, ub, swarmsize=100, seed=s, maxiter=50)
        weight_list.append(extract_weight(optx))
        print("Round {}: Completed".format(s+1))
    # Extract raw weights
    w1_list, w2_list, w3_list, w4_list = [], [], [], []
    for weight in weight_list:
        w1_list.append(weight[0])
        w2_list.append(weight[1])
        w3_list.append(weight[2])
        w4_list.append(weight[3])
    # Normalize weights
    w1_avage_norm = np.round(np.mean(w1_list), 4)
    w2_avage_norm = np.round(np.mean(w2_list), 4)
    w3_avage_norm = np.round(np.mean(w3_list), 4)
    w4_avage_norm = np.round(np.mean(w4_list), 4)
    # Export weight
    w = [w1_avage_norm, w2_avage_norm, w3_avage_norm, w4_avage_norm]
    normalized_weight_list.append(w)
    #--------------------------------------------------------------------#
    # Testing models
    test_pred_svm_mf1024   = my_svm_mf1024.fit(np.concatenate([train_mf1024, val_mf1024]), np.concatenate([train_label, val_label])).predict_proba(test_mf1024)[::,1]
    test_pred_svm_mf2048   = my_svm_mf2048.fit(np.concatenate([train_mf2048, val_mf2048]), np.concatenate([train_label, val_label])).predict_proba(test_mf2048)[::,1]
    test_pred_rf_md        = my_rf_md.fit(np.concatenate([train_md, val_md]), np.concatenate([train_label, val_label])).predict_proba(test_md)[::,1]
    test_pred_xgb_mol2vec  = my_xgb_mol2vec.fit(np.concatenate([train_mol2vec, val_mol2vec]), np.concatenate([train_label, val_label])).predict_proba(test_mol2vec)[::,1]
    test_pred_top4ensemble = (test_pred_svm_mf1024*w1_avage_norm + test_pred_svm_mf2048*w2_avage_norm + test_pred_rf_md*w3_avage_norm + test_pred_xgb_mol2vec*w4_avage_norm)
    #--------------------------------------------------------------------#
    # Calculate AUC of all models
    test_roc_auc_svm_mf1024   = roc_auc_score(test_label, test_pred_svm_mf1024)
    test_roc_auc_svm_mf2048   = roc_auc_score(test_label, test_pred_svm_mf2048)
    test_roc_auc_rf_md        = roc_auc_score(test_label, test_pred_rf_md)
    test_roc_auc_xgb_mol2vec  = roc_auc_score(test_label, test_pred_xgb_mol2vec)
    test_roc_auc_top4ensemble = roc_auc_score(test_label, test_pred_top4ensemble)
    print("meta_AUC: {}".format(np.round(test_roc_auc_top4ensemble, 4)))
    svm_mf1024_test_auc_list.append(test_roc_auc_svm_mf1024)
    svm_mf2048_test_auc_list.append(test_roc_auc_svm_mf2048)
    rf_md_test_auc_list.append(test_roc_auc_rf_md)
    xgb_test_auc_list.append(test_roc_auc_xgb_mol2vec)
    meta_test_auc_list.append(test_roc_auc_top4ensemble)
    #--------------------------------------------#
    # Export predicted probabilities of all models
    meta_pred_list.append(test_pred_top4ensemble)
    svm_mf1024_pred_list.append(test_pred_svm_mf1024)
    svm_mf2048_pred_list.append(test_pred_svm_mf2048)
    rf_md_pred_list.append(test_pred_rf_md)
    xgb_mol2vec_pred_list.append(test_pred_xgb_mol2vec)

#============================================================
mean = round(sts.mean(meta_test_auc_list), 4)
sd   = round(sts.stdev(meta_test_auc_list), 4)
ci95 = ( round((mean - 1.96*sd/math.sqrt(10)), 4), round((mean + 1.96*sd/math.sqrt(10)),4))

#============================================================
# Export results
trials = []
for i in range(1,31):
    trial = 'trial_' + str(i)
    trials.append(trial)   

outpath = "../../results/pso/allseeds/"
if os.path.isdir(outpath) is False:
    os.makedirs(outpath)
    
w_df = pd.DataFrame(normalized_weight_list).T
w_df.columns = trials
w_df.to_csv("../../results/pso/allseeds/weight_allseeds.csv", index=False)

combined_auc = pd.DataFrame([svm_mf1024_test_auc_list, svm_mf2048_test_auc_list, rf_md_test_auc_list, xgb_test_auc_list, meta_test_auc_list]).T
combined_auc.columns = ['svm_mf1024', 'svm_mf2048', 'rf_md_pred', 'xgb_mol2vec', 'meta-classifier']
combined_auc.to_csv("../../results/pso/allseeds/combined_auc_allseeds.csv", index=False)

meta_pred         = pd.DataFrame(meta_pred_list).T
meta_pred.columns = trials
meta_pred.to_csv("../../results/pso/allseeds/combined_pred_meta_top4_allseeds.csv", index=False)

svm_mf1024_pred         = pd.DataFrame(svm_mf1024_pred_list).T
svm_mf1024_pred.columns = trials
svm_mf1024_pred.to_csv("../../results/pso/allseeds/combined_pred_svm_mf1024_allseeds.csv", index=False)

svm_mf2048_pred         = pd.DataFrame(svm_mf2048_pred_list).T
svm_mf2048_pred.columns = trials
svm_mf2048_pred.to_csv("../../results/pso/allseeds/combined_pred_svm_mf2048_allseeds.csv", index=False)

rf_md_pred         = pd.DataFrame(rf_md_pred_list).T
rf_md_pred.columns = trials
rf_md_pred.to_csv("../../results/pso/allseeds/combined_pred_rf_md_allseeds.csv", index=False)

xgb_mol2vec_pred         = pd.DataFrame(xgb_mol2vec_pred_list).T
xgb_mol2vec_pred.columns = trials
xgb_mol2vec_pred.to_csv("../../results/pso/allseeds/combined_pred_xgb_mol2vec_allseeds.csv", index=False)

test_label   = pd.DataFrame(np.load("../../data/allseeds/seed_21/featurised_data/label_test.npy"), columns=['true_label'])
error_ana_df = pd.concat([svm_mf1024_pred.iloc[:,0], svm_mf2048_pred.iloc[:,0], rf_md_pred.iloc[:,0], xgb_mol2vec_pred.iloc[:,0], meta_pred.iloc[:,0], test_label], axis=1)
error_ana_df.columns = ['svm_mf1024', 'svm_mf2048', 'rf_md', 'xgb_mol2vec', 'meta', 'true_label']
error_ana_df.to_csv("../../results/pso/allseeds/error_analysis.csv", index=False)