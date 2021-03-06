# Import libraries
import pandas as pd
import os

#============================================================
# Organize results
models = ['knn', 'svm', 'rf', 'xgb']

for model in models:
    concat_df = pd.read_csv("./md_model_tuning/result/{}/{}_md_seed0.csv".format(model, model.upper()))['Metrics']
    rdkitmd   = pd.read_csv("./md_model_tuning/result/{}/{}_md_seed0.csv".format(model, model.upper()))['md']
    mf_1024   = pd.read_csv("./mf_model_tuning/result/{}/{}_mf_1024_seed0.csv".format(model, model.upper()))['mf_1024']
    mf_2048   = pd.read_csv("./mf_model_tuning/result/{}/{}_mf_2048_seed0.csv".format(model, model.upper()))['mf_2048']
    mol2vec   = pd.read_csv("./mol2vec_moldel_tuning/result/{}/{}_mol2vec_seed0.csv".format(model, model.upper()))['mol2vec']
    concat_df = pd.concat([concat_df, rdkitmd, mf_1024, mf_2048, mol2vec], axis=1)
    outpath = '../../results/model_tuning_combined_results/singlerun/'
    if os.path.isdir(outpath) is False:
        os.makedirs(outpath)
    concat_df.to_csv("../../results/model_tuning_combined_results/singlerun/{}.csv".format(model), index=False)
