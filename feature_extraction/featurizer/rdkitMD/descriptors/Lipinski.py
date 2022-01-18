import pandas as pd
from rdkit.Chem import Lipinski, Descriptors

colnames = ['NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings', 
            'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 
            'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 
            'NumSaturatedRings', 'NumValenceElectrons']
def get_Lipinski(mol):
    NHOHCount = Lipinski.NHOHCount(mol)
    NOCount = Lipinski.NOCount(mol)
    NumAliphaticCarbocycles = Lipinski.NumAliphaticCarbocycles(mol)
    NumAliphaticHeterocycles = Lipinski.NumAliphaticHeterocycles(mol)
    NumAliphaticRings = Lipinski.NumAliphaticRings(mol)
    NumAromaticCarbocycles = Lipinski.NumAromaticCarbocycles(mol)
    NumAromaticHeterocycles = Lipinski.NumAromaticHeterocycles(mol)
    NumAromaticRings = Lipinski.NumAromaticRings(mol)
    NumHAcceptors = Lipinski.NumHAcceptors(mol)
    NumHDonors = Lipinski.NumHDonors(mol)
    NumHeteroatoms = Lipinski.NumHeteroatoms(mol)
    NumRadicalElectrons = Descriptors.NumRadicalElectrons(mol)
    NumRotatableBonds = Lipinski.NumRotatableBonds(mol)
    NumSaturatedCarbocycles = Lipinski.NumSaturatedCarbocycles(mol)
    NumSaturatedHeterocycles = Lipinski.NumSaturatedHeterocycles(mol)
    NumSaturatedRings = Lipinski.NumSaturatedRings(mol)
    NumValenceElectrons = Descriptors.NumValenceElectrons(mol)
    LipinskiCount = [NHOHCount, NOCount, NumAliphaticCarbocycles, NumAliphaticHeterocycles, NumAliphaticRings, NumAromaticCarbocycles, NumAromaticHeterocycles, NumAromaticRings, NumHAcceptors, NumHDonors, NumHeteroatoms, NumRadicalElectrons, NumRotatableBonds, NumSaturatedCarbocycles, NumSaturatedHeterocycles, NumSaturatedRings, NumValenceElectrons] 
    LipinskiCount_df = pd.DataFrame(LipinskiCount).T
    LipinskiCount_df.columns = colnames
    return LipinskiCount_df

