# Import libraries
import pandas as pd

#============================================================
# Process rawfile of RDKit MD 
for i in range(10):
    test  = pd.read_csv("../data/ten_seeds/seed_{}/rdkit_md/result.csv".format(i)).iloc[::,0:-1]
    train = pd.read_csv("../data/ten_seeds/seed_{}/rdkit_md/result (1).csv".format(i)).iloc[::,0:-1]
    val   = pd.read_csv("../data/ten_seeds/seed_{}/rdkit_md/result (2).csv".format(i)).iloc[::,0:-1]
    test.to_csv("../data/ten_seeds/seed_{}/rdkit_md/rdkit_md_test.csv".format(i), index=False)
    train.to_csv("../data/ten_seeds/seed_{}/rdkit_md/rdkit_md_train.csv".format(i), index=False)
    val.to_csv("../data/ten_seeds/seed_{}/rdkit_md/rdkit_md_val.csv".format(i), index=False)
