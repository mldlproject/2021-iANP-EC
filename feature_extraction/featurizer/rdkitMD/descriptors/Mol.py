import pandas as pd
from rdkit.Chem import Crippen, Descriptors, MolSurf, Lipinski, GraphDescriptors
from rdkit.Chem.EState import EState

colnames = ['Kappa1', 'Kappa2', 'Kappa3',
            'ExactMolWt', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt', 'Ipc',
            'LabuteASA', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge', 
            'MolLogP', 'MolMR', 'MolWt']

def get_Mol(mol):
    ExactMolWt = Descriptors.ExactMolWt(mol)
    FractionCSP3 = Lipinski.FractionCSP3(mol)
    HallKierAlpha = GraphDescriptors.HallKierAlpha(mol)
    HeavyAtomCount = Descriptors.HeavyAtomCount(mol)
    HeavyAtomMolWt = Descriptors.HeavyAtomMolWt(mol)
    Ipc = Descriptors.Ipc(mol)
    Mol1 = [ExactMolWt, FractionCSP3, HallKierAlpha, HeavyAtomCount, HeavyAtomMolWt, Ipc]
    
    Kappa1 = GraphDescriptors.Kappa1(mol)
    Kappa2 = GraphDescriptors.Kappa2(mol)
    Kappa3 = GraphDescriptors.Kappa3(mol)
    Mol2 = [Kappa1, Kappa2, Kappa3]
    
    LabuteASA = MolSurf.LabuteASA(mol)
    MaxAbsEStateIndex = EState.MaxAbsEStateIndex(mol)
    MaxAbsPartialCharge = Descriptors.MaxAbsPartialCharge(mol)
    MaxEStateIndex = EState.MaxEStateIndex(mol)
    MaxPartialCharge = Descriptors.MaxPartialCharge(mol)
    MinAbsEStateIndex = EState.MinAbsEStateIndex(mol)
    MinAbsPartialCharge = Descriptors.MinAbsPartialCharge(mol)
    MinEStateIndex = EState.MinEStateIndex(mol)
    MinPartialCharge = Descriptors.MinPartialCharge(mol)
    Mol3 = [LabuteASA, MaxAbsEStateIndex, MaxAbsPartialCharge, MaxEStateIndex, MaxPartialCharge, MinAbsEStateIndex, MinAbsPartialCharge, MinEStateIndex, MinPartialCharge]
    
    MolLogP = Crippen.MolLogP(mol)
    MolMR = Crippen.MolMR(mol)
    MolWt = Descriptors.MolWt(mol)
    Mol4 = [ MolLogP, MolMR, MolWt]
    
    Mol = Mol1 + Mol2 + Mol3 + Mol4
    Mol_df = pd.DataFrame(Mol).T
    Mol_df.columns = colnames
    return Mol_df