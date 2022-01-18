import pandas as pd
from rdkit.Chem.EState import EState_VSA
from rdkit.Chem import Descriptors

colnames = ['TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9']
def get_VSA_EState(mol):
    TPSA = Descriptors.TPSA(mol)
    VSA_EState1  = EState_VSA.VSA_EState1(mol)
    VSA_EState10 = EState_VSA.VSA_EState10(mol)
    VSA_EState2  = EState_VSA.VSA_EState2(mol)
    VSA_EState3  = EState_VSA.VSA_EState3(mol)
    VSA_EState4  = EState_VSA.VSA_EState4(mol)
    VSA_EState5  = EState_VSA.VSA_EState5(mol)
    VSA_EState6  = EState_VSA.VSA_EState6(mol)
    VSA_EState7  = EState_VSA.VSA_EState7(mol)
    VSA_EState8  = EState_VSA.VSA_EState8(mol)
    VSA_EState9  = EState_VSA.VSA_EState9(mol)
    VSA_EState = [TPSA, VSA_EState1, VSA_EState10, VSA_EState2, VSA_EState3, VSA_EState4, VSA_EState5, VSA_EState6, VSA_EState7, VSA_EState8, VSA_EState9]#11
    VSA_EState_df = pd.DataFrame(VSA_EState).T
    VSA_EState_df.columns = colnames
    return VSA_EState_df