# Import Library
import numpy as np
from tuning_func import knn_tuning, svm_tuning, xgb_tuning, rf_tuning
import time
#--------------------------------------------------------------------#
# TRAINING MODEL                                                     #
#--------------------------------------------------------------------#
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

for i in range(10):
    fea_list = ['1024', '2048']
    for fea in fea_list:
        X_data = np.load("../../../data/allseeds/seed_{}/featurised_data/fp{}_train.npy".format(i, fea))
        y_data = np.load("../../../data/allseeds/seed_{}/featurised_data/label_train.npy".format(i)) 
        start_knn = time.time()
        knn_tuning(X_data, y_data, tag="mf_{}".format(fea), para=knn_para, seed=i)
        end_knn = time.time()
        print("KNN Time:{} seconds".format(end_knn-start_knn))
        start_svm = time.time()
        svm_tuning(X_data, y_data, tag="mf_{}".format(fea), para=svm_para, seed=i)
        end_svm = time.time()
        print("SVM Time:{} seconds".format(end_svm-start_svm))
        start_xgb = time.time()
        xgb_tuning(X_data, y_data, tag="mf_{}".format(fea), para=xgb_para, seed=i)
        end_xgb = time.time()
        print("XGB Time:{} seconds".format(end_xgb-start_xgb))
        start_rf = time.time()
        rf_tuning(X_data, y_data, tag="mf_{}".format(fea), para=rf_para, seed=i)
        end_rf = time.time()
        print("RF Time:{} seconds".format(end_rf-start_rf))
    