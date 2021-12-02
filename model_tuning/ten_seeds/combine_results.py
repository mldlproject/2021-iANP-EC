# Import libraries
import pandas as pd
import os

#============================================================
# Organize results
outpath = "../../results/model_tuning_combined_results/ten_seeds/"
if os.path.isdir(outpath) is False:
    os.makedirs(outpath)
    
models = ['knn', 'svm', 'rf', 'xgb']
        
# md results
for m in models:
    concat_df = pd.read_csv("./md_model_tuning/ten_seeds_results/seed_0/result/{}/{}_md_seed0.csv".format(m, m.upper()))['Metrics']
    for i in range(10):
        df   = pd.read_csv("./md_model_tuning/ten_seeds_results/seed_{}/result/{}/{}_md_seed{}.csv".format(i, m, m.upper(), i))['md']
        df.name = 'seed_' + str(i) 
        concat_df = pd.concat([concat_df, df], axis=1)
    concat_df['average'] = concat_df.iloc[:,1:].sum(axis=1)/10
    concat_df['sd'] = concat_df.iloc[:,1:].std(axis=1)
    concat_df.to_csv("../../results/model_tuning_combined_results/ten_seeds/{}_md.csv".format(m.upper()), index=False)
    
# mf1024 results
for m in models:
    concat_df = pd.read_csv("./mf_model_tuning/ten_seeds_results/seed_0/result/{}/{}_mf_1024_seed0.csv".format(m, m.upper()))['Metrics']
    for i in range(10):
        df   = pd.read_csv("./mf_model_tuning/ten_seeds_results/seed_{}/result/{}/{}_mf_1024_seed{}.csv".format(i, m, m.upper(), i))['mf_1024']
        df.name = 'seed_' + str(i) 
        concat_df = pd.concat([concat_df, df], axis=1)
    concat_df['average'] = concat_df.iloc[:,1:].sum(axis=1)/10
    concat_df['sd'] = concat_df.iloc[:,1:].std(axis=1)
    concat_df.to_csv("../../results/model_tuning_combined_results/ten_seeds/{}_mf1024.csv".format(m.upper()), index=False)
    
# mf2048 results
for m in models:
    concat_df = pd.read_csv("./mf_model_tuning/ten_seeds_results/seed_0/result/{}/{}_mf_2048_seed0.csv".format(m, m.upper()))['Metrics']
    for i in range(10):
        df   = pd.read_csv("./mf_model_tuning/ten_seeds_results/seed_{}/result/{}/{}_mf_2048_seed{}.csv".format(i, m, m.upper(), i))['mf_2048']
        df.name = 'seed_' + str(i) 
        concat_df = pd.concat([concat_df, df], axis=1)
    concat_df['average'] = concat_df.iloc[:,1:].sum(axis=1)/10
    concat_df['sd'] = concat_df.iloc[:,1:].std(axis=1)
    concat_df.to_csv("../../results/model_tuning_combined_results/ten_seeds/{}_mf2048.csv".format(m.upper()), index=False)
    
# mol2vec results
for m in models:
    concat_df = pd.read_csv("./mol2vec_model_tuning/ten_seeds_results/seed_0/result/{}/{}_mol2vec_seed0.csv".format(m, m.upper()))['Metrics']
    for i in range(10):
        df   = pd.read_csv("./mol2vec_model_tuning/ten_seeds_results/seed_{}/result/{}/{}_mol2vec_seed{}.csv".format(i, m, m.upper(), i))['mol2vec']
        df.name = 'seed_' + str(i) 
        concat_df = pd.concat([concat_df, df], axis=1)
    concat_df['average'] = concat_df.iloc[:,1:].sum(axis=1)/10
    concat_df['sd'] = concat_df.iloc[:,1:].std(axis=1)
    concat_df.to_csv("../../results/model_tuning_combined_results/ten_seeds/{}_mol2vec.csv".format(m.upper()), index=False)
    
# base classifiers
for m in models:
    concat_df   = pd.read_csv("./combined_results/{}_md.csv".format(m.upper()))['Metrics']
    md          = pd.read_csv("./combined_results/{}_md.csv".format(m.upper()))[['average', 'sd']]
    mf1024      = pd.read_csv("./combined_results/{}_mf1024.csv".format(m.upper()))[['average', 'sd']]
    mf2048      = pd.read_csv("./combined_results/{}_mf2048.csv".format(m.upper()))[['average', 'sd']]
    mol2vec     = pd.read_csv("./combined_results/{}_mol2vec.csv".format(m.upper()))[['average', 'sd']]
    concat_df   = pd.concat([concat_df, md, mf1024, mf2048, mol2vec], axis=1)
    concat_df.columns = ['Metrics', 'mean_md', 'sd_md', 'mean_mf1024', 'sd_mf1024', 'mean_2048', 'sd_2048', 'mean_mol2vec', 'sd_mol2vec']
    concat_df.to_csv("../../results/model_tuning_combined_results/ten_seeds/{}_all.csv".format(m.upper()), index=False)

    