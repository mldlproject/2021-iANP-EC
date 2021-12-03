# Import libraries
import pandas as pd
import numpy as np
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import statistics as sts
import math

#============================================================
# Remove Low-variance Features
from sklearn.feature_selection import VarianceThreshold
threshold = (.95 * (1 - .95))

# Normalize Features
from sklearn.preprocessing import MinMaxScaler

#============================================================
result_list=[]
for i in range(10):
    train_md      = np.load("../../data/ten_seeds/seed_{}/featurised_data/rdkit_md_train.npy".format(i))
    train_label   = np.load("../../data/ten_seeds/seed_{}/featurised_data/label_train.npy".format(i))
    val_md        = np.load("../../data/ten_seeds/seed_{}/featurised_data/rdkit_md_val.npy".format(i))
    val_label     = np.load("../../data/ten_seeds/seed_{}/featurised_data/label_val.npy".format(i))
    test_md       = np.load("../../data/ten_seeds/seed_{}/featurised_data/rdkit_md_test.npy".format(i))
    test_label    = np.load("../../data/ten_seeds/seed_{}/featurised_data/label_test.npy".format(i))
    metric_df     = pd.read_csv("../../models_tuning/ten_seeds/md_model_tuning/ten_seeds_results/seed_{}/result/rf/RF_md_seed{}.csv".format(i, i)).iloc[11:,1].reset_index(drop=True)
    my_rf_md = make_pipeline(MinMaxScaler(),
                             VarianceThreshold(),
                             RandomForestClassifier(random_state=42,
                                                    n_estimators=100,
                                                    max_depth=int(metric_df[0]),
                                                    max_features=metric_df[1],
                                                    min_samples_split=int(metric_df[2])))
    test_pred_md     = my_rf_md.fit(np.concatenate([train_md, val_md]), np.concatenate([train_label, val_label])).predict_proba(test_md)[::,1]
    test_roc_auc_md  = roc_auc_score(test_label, test_pred_md)
    result_list.append(test_roc_auc_md)

mean = round(sts.mean(result_list),4)
sd   = round(sts.stdev(result_list), 4)
ci95 = ( round((mean - 1.96*sd/math.sqrt(10)), 4), round((mean + 1.96*sd/math.sqrt(10)),4))

