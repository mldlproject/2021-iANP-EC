import pandas as pd
from rdkit.Chem import MolSurf

colnames = ['PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9']
def get_PEOE_VSA(mol):
    PEOE_VSA1 = MolSurf.PEOE_VSA1(mol, y=0)
    PEOE_VSA10 = MolSurf.PEOE_VSA10(mol, y=9)
    PEOE_VSA11 = MolSurf.PEOE_VSA11(mol, y=10)
    PEOE_VSA12 = MolSurf.PEOE_VSA12(mol, y=11)
    PEOE_VSA13 = MolSurf.PEOE_VSA13(mol, y=12)
    PEOE_VSA14 = MolSurf.PEOE_VSA14(mol, y=13)
    PEOE_VSA2 = MolSurf.PEOE_VSA2(mol, y=1)
    PEOE_VSA3 = MolSurf.PEOE_VSA3(mol, y=2)
    PEOE_VSA4 = MolSurf.PEOE_VSA4(mol, y=3)
    PEOE_VSA5 = MolSurf.PEOE_VSA5(mol, y=4)
    PEOE_VSA6 = MolSurf.PEOE_VSA6(mol, y=5)
    PEOE_VSA7 = MolSurf.PEOE_VSA7(mol, y=6)
    PEOE_VSA8 = MolSurf.PEOE_VSA8(mol, y=7)
    PEOE_VSA9 = MolSurf.PEOE_VSA9(mol, y=8)
    PEOE_VSA = [PEOE_VSA1, PEOE_VSA10, PEOE_VSA11, PEOE_VSA12, PEOE_VSA13, PEOE_VSA14, PEOE_VSA2, PEOE_VSA3, PEOE_VSA4, PEOE_VSA5, PEOE_VSA6, PEOE_VSA7, PEOE_VSA8, PEOE_VSA9]#14
    PEOE_VSA_df = pd.DataFrame(PEOE_VSA).T
    PEOE_VSA_df.columns = colnames
    return PEOE_VSA_df

