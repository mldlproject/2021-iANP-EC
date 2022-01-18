import pandas as pd
from rdkit.Chem import MolSurf

colnames = ['SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9']
def get_SlogP_VSA(mol):
    SlogP_VSA1  = MolSurf.SlogP_VSA1(mol, y=0)
    SlogP_VSA10 = MolSurf.SlogP_VSA10(mol, y=9)
    SlogP_VSA11 = MolSurf.SlogP_VSA11(mol, y=10)
    SlogP_VSA12 = MolSurf.SlogP_VSA12(mol, y=11)
    SlogP_VSA2  = MolSurf.SlogP_VSA2(mol, y=1)
    SlogP_VSA3  = MolSurf.SlogP_VSA3(mol, y=2)
    SlogP_VSA4  = MolSurf.SlogP_VSA4(mol, y=3)
    SlogP_VSA5  = MolSurf.SlogP_VSA5(mol, y=4)
    SlogP_VSA6  = MolSurf.SlogP_VSA6(mol, y=5)
    SlogP_VSA6  = MolSurf.SlogP_VSA6(mol, y=5)
    SlogP_VSA7  = MolSurf.SlogP_VSA7(mol, y=6)
    SlogP_VSA8  = MolSurf.SlogP_VSA8(mol, y=7)
    SlogP_VSA9  = MolSurf.SlogP_VSA9(mol, y=8)
    SlogP_VSA = [SlogP_VSA1, SlogP_VSA10, SlogP_VSA11, SlogP_VSA12, SlogP_VSA2, SlogP_VSA3, SlogP_VSA4, SlogP_VSA5, SlogP_VSA6, SlogP_VSA7, SlogP_VSA8, SlogP_VSA9]#12
    SlogP_VSA_df = pd.DataFrame(SlogP_VSA).T
    SlogP_VSA_df.columns = colnames
    return SlogP_VSA_df