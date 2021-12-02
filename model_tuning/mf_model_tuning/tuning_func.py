# Import Library
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import os
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from utils import printPerformance

#--------------------------------------------------------------------#
# Remove Low-variance Features
from sklearn.feature_selection import VarianceThreshold
threshold = (.95 * (1 - .95))

#--------------------------------------------------------------------#
# TUNING KNN MODEL                                                   #
#--------------------------------------------------------------------#
def knn_tuning(X_train, y_train, scoring = 'roc_auc', nfold=5, tag=None, seed=0, para=None):
    X_train_, y_train_ = X_train, y_train
    if tag is not None:
        tag_ = tag
    seed_     = seed
    scoring_  = scoring
    nfold_    = nfold
    concat_df = pd.DataFrame(['AUC-ROC', 'AUC-PR','ACC', 'BA', 'MCC', 
                              'SN/RE', 'SP', 'PR', 'F1', 'CK', 
                              'cross_validated',
                              'best_n_neighbors'], columns= ["Metrics"]) 
    #=====================================#
    # Set Up Parameter
    if para == None:
        my_n_neighbors = np.arange(3,21)
    else:
        my_n_neighbors = para[0]
    #=====================================# 
    my_classifier = make_pipeline(VarianceThreshold(), 
                                  KNeighborsClassifier())
    #=====================================#
    # GridsearchCV
    my_parameters_grid = {'n_neighbors': my_n_neighbors}
    my_new_parameters_grid = {'kneighborsclassifier__' + key: my_parameters_grid[key] for key in my_parameters_grid}
    grid_cv = GridSearchCV(my_classifier, 
                           my_new_parameters_grid, 
                           scoring=scoring_,
                           n_jobs= 1, 
                           cv = StratifiedKFold(n_splits=nfold_, shuffle=True, random_state=42), 
                           return_train_score=True)
    grid_cv.fit(X_train_, y_train_)
    #=====================================#
    # Create Regressor uing Best Parameters (Use only one option at each run)
    best_n_neighbors = grid_cv.best_params_['kneighborsclassifier__n_neighbors']
    cross_validated  = grid_cv.best_score_      
    best_para_set    = [cross_validated, best_n_neighbors]
    #=====================================#
    my_best_classifier = make_pipeline(VarianceThreshold(),
                                       KNeighborsClassifier(n_neighbors=best_n_neighbors))
    #=====================================#
    # Testing on train data
    my_best_classifier.fit(X_train_, y_train_)
    y_pred = my_best_classifier.predict(X_train_)
    y_prob = my_best_classifier.predict_proba(X_train_)[::,1]
    #=====================================#
    pred_path = './pred/knn/'
    if os.path.isdir(pred_path) is False:
        os.makedirs(pred_path)
    pd.DataFrame.to_csv(pd.DataFrame(y_prob, columns = ['Predicted Probability']), './pred/knn/y_prob_KNN_{}_seed{}.csv'.format(tag_, seed_), index=False)
    #=====================================#
    # Evaluation
    x         = printPerformance(y_train, y_prob)
    print(x)
    x_list    = list(x)
    x_list    = x_list + best_para_set
    new_df    = pd.DataFrame(x_list, columns = [tag_])
    concat_df = pd.concat([concat_df, new_df], axis=1)
    result_path = './result/knn/'
    if os.path.isdir(result_path) is False:
        os.makedirs(result_path)
    pd.DataFrame.to_csv(concat_df, './result/knn/KNN_{}_seed{}.csv'.format(tag_, seed_), index=False)

#--------------------------------------------------------------------#
# TUNING SVM MODEL                                                   #
#--------------------------------------------------------------------#
def svm_tuning(X_train, y_train, scoring = 'roc_auc', nfold=5, tag=None, seed=0, para=None):
    X_train_, y_train_ = X_train, y_train
    if tag is not None:
        tag_ = tag
    seed_     = seed
    scoring_  = scoring
    nfold_    = nfold
    concat_df = pd.DataFrame(['AUC-ROC', 'AUC-PR','ACC', 'BA', 
                              'SN/RE', 'SP', 'PR', 'MCC', 'F1', 'CK', 
                              'cross_validated',
                              'best_C', 'best_gamma'], columns= ["Metrics"]) 
    #=====================================#
    # Set Up Parameter
    if para == None:
        my_C     = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100]
        my_gamma = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.6, 0.7, 0.8, 0.8, 0.9, 1]
    else:
        my_C, my_gamma = para[0], para[1]
    #=====================================# 
    my_classifier = make_pipeline(VarianceThreshold(), 
                                  SVC())
    #=====================================#
    # GridsearchCV
    my_parameters_grid = {'C': my_C, 'gamma': my_gamma}
    my_new_parameters_grid = {'svc__' + key: my_parameters_grid[key] for key in my_parameters_grid}
    grid_cv = GridSearchCV(my_classifier, 
                           my_new_parameters_grid, 
                           scoring=scoring_,
                           n_jobs= 1, 
                           cv = StratifiedKFold(n_splits=nfold_, shuffle=True, random_state=42), 
                           return_train_score=True)
    grid_cv.fit(X_train_, y_train_)
    #=====================================#
    # Create Regressor uing Best Parameters (Use only one option at each run)
    best_C           = grid_cv.best_params_['svc__C']
    best_gamma       = grid_cv.best_params_['svc__gamma']
    cross_validated  = grid_cv.best_score_   
    best_para_set    = [cross_validated, best_C, best_gamma]
    #=====================================#
    my_best_classifier = make_pipeline(VarianceThreshold(),
                                       SVC(C=best_C, gamma=best_gamma, probability=True))
    #=====================================#
    # Testing on train data
    my_best_classifier.fit(X_train_, y_train_)
    y_pred = my_best_classifier.predict(X_train_)
    y_prob = my_best_classifier.predict_proba(X_train_)[::,1]
    #=====================================#
    pred_path = './pred/svm/'
    if os.path.isdir(pred_path) is False:
        os.makedirs(pred_path)
    pd.DataFrame.to_csv(pd.DataFrame(y_prob, columns = ['Predicted Probability']), './pred/svm/y_prob_SVM_{}_seed{}.csv'.format(tag_, seed_), index=False)
    #=====================================#
    # Evaluation
    x         = printPerformance(y_train, y_prob)
    print(x)
    x_list    = list(x)
    x_list    = x_list + best_para_set
    new_df    = pd.DataFrame(x_list, columns = [tag_])
    concat_df = pd.concat([concat_df, new_df], axis=1)
    result_path = './result/svm/'
    if os.path.isdir(result_path) is False:
        os.makedirs(result_path)
    pd.DataFrame.to_csv(concat_df, './result/svm/SVM_{}_seed{}.csv'.format(tag_, seed_), index=False)
    
#--------------------------------------------------------------------#
# TUNING XGB MODEL                                                   #
#--------------------------------------------------------------------#
def xgb_tuning(X_train, y_train, scoring = 'roc_auc', nfold=5, tag=None, seed=0, para=None):
    X_train_, y_train_ = X_train, y_train
    if tag is not None:
        tag_ = tag
    seed_     = seed
    scoring_  = scoring
    nfold_    = nfold
    concat_df = pd.DataFrame(['AUC-ROC', 'AUC-PR','ACC', 'BA', 
                              'SN/RE', 'SP', 'PR', 'MCC', 'F1', 'CK', 
                              'cross_validated',
                              'best_max_depth', 'best_colsample_bytree', 'best_learning_rate'], columns= ["Metrics"]) 
    #=====================================#
    # Set Up Parameter
    if para == None:        
        my_n_estimators     = 200
        my_learning_rate    = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1]
        my_max_depth        = np.arange(3, 10)
        my_colsample_bytree = np.arange(0.2, 0.95, 0.05)
    else:
        my_n_estimators, my_learning_rate, my_max_depth, my_colsample_bytree = para[0], para[1], para[2], para[3]
    #=====================================# 
    my_classifier = make_pipeline(VarianceThreshold(), 
                                  XGBClassifier(random_state=42, n_estimators=my_n_estimators))
    #=====================================#
    # GridsearchCV
    my_parameters_grid = {'max_depth': my_max_depth, 'learning_rate': my_learning_rate, 'colsample_bytree': my_colsample_bytree}
    my_new_parameters_grid = {'xgbclassifier__' + key: my_parameters_grid[key] for key in my_parameters_grid}
    grid_cv = GridSearchCV(my_classifier, 
                           my_new_parameters_grid, 
                           scoring=scoring_,
                           n_jobs= 1, 
                           cv = StratifiedKFold(n_splits=nfold_, shuffle=True, random_state=42), 
                           return_train_score=True)
    grid_cv.fit(X_train_, y_train_)
    #=====================================#
    # Create Regressor uing Best Parameters (Use only one option at each run)
    best_max_depth        = grid_cv.best_params_['xgbclassifier__max_depth']
    best_colsample_bytree = grid_cv.best_params_['xgbclassifier__colsample_bytree']
    best_learning_rate    = grid_cv.best_params_['xgbclassifier__learning_rate']
    cross_validated       = grid_cv.best_score_  
    best_para_set         = [cross_validated, best_max_depth, best_colsample_bytree, best_learning_rate]
    #=====================================#
    my_best_classifier = make_pipeline(VarianceThreshold(),
                                       XGBClassifier(random_state=42, 
                                                     n_estimators     = my_n_estimators,
                                                     max_depth        = best_max_depth,
                                                     colsample_bytree = best_colsample_bytree,
                                                     learning_rate    = best_learning_rate))
    #=====================================#
    # Testing on train data
    my_best_classifier.fit(X_train_, y_train_)
    y_pred = my_best_classifier.predict(X_train_)
    y_prob = my_best_classifier.predict_proba(X_train_)[::,1]
    #=====================================#
    pred_path = './pred/xgb/'
    if os.path.isdir(pred_path) is False:
        os.makedirs(pred_path)
    pd.DataFrame.to_csv(pd.DataFrame(y_prob, columns = ['Predicted Probability']), './pred/xgb/y_prob_XGB_{}_seed{}.csv'.format(tag_, seed_), index=False)
    #=====================================#
    # Evaluation
    x         = printPerformance(y_train, y_prob)
    print(x)
    x_list    = list(x)
    x_list    = x_list + best_para_set
    new_df    = pd.DataFrame(x_list, columns = [tag_])
    concat_df = pd.concat([concat_df, new_df], axis=1)
    result_path = './result/xgb/'
    if os.path.isdir(result_path) is False:
        os.makedirs(result_path)
    pd.DataFrame.to_csv(concat_df, './result/xgb/XGB_{}_seed{}.csv'.format(tag_, seed_), index=False)
    
#--------------------------------------------------------------------#
# TUNING RF MODEL                                                    #
#--------------------------------------------------------------------#
def rf_tuning(X_train, y_train, scoring = 'roc_auc', nfold=5, tag=None, seed=0, para=None):
    X_train_, y_train_ = X_train, y_train
    if tag is not None:
        tag_ = tag
    seed_     = seed
    scoring_  = scoring
    nfold_    = nfold
    concat_df = pd.DataFrame(['AUC-ROC', 'AUC-PR','ACC', 'BA', 
                              'SN/RE', 'SP', 'PR', 'MCC', 'F1', 'CK', 
                              'cross_validated',
                              'best_max_depth', 'best_max_features', 'best_min_samples_split'], columns= ["Metrics"]) 
    #=====================================#
    # Set Up Parameter
    if para == None:
        my_n_estimators      = 200
        my_max_depth         = np.arange(2, 10)
        my_max_features      = np.arange(0.2, 0.95, 0.05)
        my_min_samples_split = np.arange(2, 10)
    else:
        my_n_estimators, my_max_depth, my_max_features, my_min_samples_split = para[0], para[1], para[2], para[3]
    #=====================================# 
    my_classifier = make_pipeline(VarianceThreshold(), 
                                  RandomForestClassifier(random_state=42, n_estimators=my_n_estimators))
    #=====================================#
    # GridsearchCV
    my_parameters_grid = {'max_depth': my_max_depth, 'max_features': my_max_features, 'min_samples_split': my_min_samples_split}
    my_new_parameters_grid = {'randomforestclassifier__' + key: my_parameters_grid[key] for key in my_parameters_grid}
    grid_cv = GridSearchCV(my_classifier, 
                           my_new_parameters_grid, 
                           scoring=scoring_,
                           n_jobs= 1, 
                           cv = StratifiedKFold(n_splits=nfold_, shuffle=True, random_state=42), 
                           return_train_score=True)
    grid_cv.fit(X_train_, y_train_)
    #=====================================#
    # Create Regressor uing Best Parameters (Use only one option at each run)
    best_max_depth         = grid_cv.best_params_['randomforestclassifier__max_depth']
    best_max_features      = grid_cv.best_params_['randomforestclassifier__max_features']
    best_min_samples_split = grid_cv.best_params_['randomforestclassifier__min_samples_split']
    cross_validated        = grid_cv.best_score_  
    best_para_set          = [cross_validated, best_max_depth, best_max_features, best_min_samples_split]
    #=====================================#
    my_best_classifier = make_pipeline(VarianceThreshold(),
                                       RandomForestClassifier(random_state      = 42,
                                                              n_estimators      = my_n_estimators, 
                                                              max_depth         = best_max_depth,
                                                              max_features      = best_max_features,
                                                              min_samples_split = best_min_samples_split))  
    #=====================================#
    # Testing on train data
    my_best_classifier.fit(X_train_, y_train_)
    y_pred = my_best_classifier.predict(X_train_)
    y_prob = my_best_classifier.predict_proba(X_train_)[::,1]
    #=====================================#
    pred_path = './pred/rf/'
    if os.path.isdir(pred_path) is False:
        os.makedirs(pred_path)
    pd.DataFrame.to_csv(pd.DataFrame(y_prob, columns = ['Predicted Probability']), './pred/rf/y_prob_RF_{}_seed{}.csv'.format(tag_, seed_), index=False)
    #=====================================#
    # Evaluation
    x         = printPerformance(y_train, y_prob)
    print(x)
    x_list    = list(x)
    x_list    = x_list + best_para_set
    new_df    = pd.DataFrame(x_list, columns = [tag_])
    concat_df = pd.concat([concat_df, new_df], axis=1)
    result_path = './result/rf/'
    if os.path.isdir(result_path) is False:
        os.makedirs(result_path)
    pd.DataFrame.to_csv(concat_df, './result/rf/RF{}_seed{}.csv'.format(tag_, seed_), index=False)
