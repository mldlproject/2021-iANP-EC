import pandas as pd
from rdkit.Chem.EState.EState_VSA import *


colnames = ['EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2', 'EState_VSA3',
            'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9']
def get_EState_VSA(mol):
    EState_VSA1_  = EState_VSA1(mol)
    EState_VSA10_ = EState_VSA10(mol)
    EState_VSA11_ = EState_VSA11(mol)
    EState_VSA2_  = EState_VSA2(mol)
    EState_VSA3_  = EState_VSA3(mol)
    EState_VSA4_  = EState_VSA4(mol)
    EState_VSA5_  = EState_VSA5(mol)
    EState_VSA6_  = EState_VSA6(mol)
    EState_VSA7_  = EState_VSA7(mol)
    EState_VSA8_  = EState_VSA8(mol)
    EState_VSA9_  = EState_VSA9(mol)
    EState_VSA = [EState_VSA1_, EState_VSA10_, EState_VSA11_, EState_VSA2_, EState_VSA3_, EState_VSA4_, EState_VSA5_, EState_VSA6_, EState_VSA7_, EState_VSA8_, EState_VSA9_]
    EState_VSA_df = pd.DataFrame(EState_VSA).T
    EState_VSA_df.columns = colnames
    return EState_VSA_df