import pandas as pd
from rdkit.Chem import MolSurf

colnames = ['SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9']
def get_SMR_VSA(mol):
    SMR_VSA1  = MolSurf.SMR_VSA1(mol, y=0)
    SMR_VSA10 = MolSurf.SMR_VSA10(mol, y=9)
    SMR_VSA2  = MolSurf.SMR_VSA2(mol, y=1)
    SMR_VSA3  = MolSurf.SMR_VSA3(mol, y=2)
    SMR_VSA4  = MolSurf.SMR_VSA4(mol, y=3)
    SMR_VSA5  = MolSurf.SMR_VSA5(mol, y=4)
    SMR_VSA6  = MolSurf.SMR_VSA6(mol, y=5)
    SMR_VSA7  = MolSurf.SMR_VSA7(mol, y=6)
    SMR_VSA8  = MolSurf.SMR_VSA8(mol, y=7)
    SMR_VSA9  = MolSurf.SMR_VSA9(mol, y=8)
    SMR_VSA = [SMR_VSA1, SMR_VSA10, SMR_VSA2, SMR_VSA3, SMR_VSA4, SMR_VSA5, SMR_VSA6, SMR_VSA7, SMR_VSA8, SMR_VSA9]#10
    SMR_VSA_df = pd.DataFrame(SMR_VSA).T
    SMR_VSA_df.columns = colnames
    return SMR_VSA_df