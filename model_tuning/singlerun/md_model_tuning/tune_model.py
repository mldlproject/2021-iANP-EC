# Import Library
import pandas as pd
import numpy as np
from tuning_func import knn_tuning, svm_tuning, xgb_tuning, rf_tuning
import time
#--------------------------------------------------------------------#
# TRAINING MODEL                                                     #
#--------------------------------------------------------------------#
#Define parameter
X_data = np.load('../../../data/singlerun/featurised_data/rdkit_md_train.npy')
y_data = np.load('../../../data/singlerun/featurised_data/label_train.npy')

#Set up parameters
my_n_neighbors = np.arange(3,16,2)
knn_para = [my_n_neighbors]

my_C     = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
my_gamma = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
svm_para = [my_C, my_gamma]

my_n_estimators_xgb     = 100
my_learning_rate_xgb    = [0.001, 0.01, 0.1]
my_max_depth_xgb        = np.arange(3, 10)
my_colsample_bytree_xgb = np.arange(0.2, 0.95, 0.1)
xgb_para = [my_n_estimators_xgb, my_learning_rate_xgb, my_max_depth_xgb, my_colsample_bytree_xgb]

my_n_estimators_rf      = 100
my_max_depth_rf         = np.arange(3, 10)
my_max_features_rf      = np.arange(0.2, 0.95, 0.1)
my_min_samples_split_rf = np.arange(2, 6)
rf_para = [my_n_estimators_rf, my_max_depth_rf, my_max_features_rf, my_min_samples_split_rf]
#Training

start_knn = time.time()
knn_tuning(X_data, y_data, tag="md", para=knn_para)
end_knn = time.time()
print("KNN Time:{} seconds".format(end_knn-start_knn))


start_svm = time.time()
svm_tuning(X_data, y_data, tag="md", para=svm_para)
end_svm = time.time()
print("SVM Time:{} seconds".format(end_svm-start_svm))


start_xgb = time.time()
xgb_tuning(X_data, y_data, tag="md", para=xgb_para)
end_xgb = time.time()
print("XGB Time:{} seconds".format(end_xgb-start_xgb))


start_rf = time.time()
rf_tuning(X_data, y_data, tag="md", para=rf_para)
end_rf = time.time()
print("RF Time:{} seconds".format(end_rf-start_rf))

