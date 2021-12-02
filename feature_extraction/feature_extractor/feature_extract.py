import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from .mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

#========================================================================================#
# Check SMILES validity
#========================================================================================#
def SMILES_check(SMILES):
    none_index, smiles_checked = [], []
    _SMILES = SMILES
    #--------------------------------------------------------
    if isinstance(_SMILES, pd.DataFrame):
        smiles_list = _SMILES[0].tolist()
    else:
        smiles_list = _SMILES #list type
    #--------------------------------------------------------
    for i in range(0, len(_SMILES)):
        smiles = smiles_list[i]
        mol = Chem.MolFromSmiles(smiles)
        if mol == None:
            none_index.append(i)
        else:
            smiles_checked.append(smiles)
    #--------------------------------------------------------
    if len(none_index) == 0:
        return smiles_checked
    else:
        #print("SMILES {} error, please check!".format(str(none_index[0])))
        return smiles_checked, none_index

#========================================================================================#
# Extract Morgan fingerprint
#========================================================================================#
def fp_extract(SMILES, bit_type=1024):
    dataX = []
    _SMILES = SMILES
    #--------------------------------------------------------
    if isinstance(_SMILES, pd.DataFrame):
        smiles_list = _SMILES[0].tolist()
    else:
        smiles_list = _SMILES #list type
    #--------------------------------------------------------
    for i in range(0, len(_SMILES)):
        smiles = smiles_list[i]
        mol = Chem.MolFromSmiles(smiles)
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits = bit_type))
        dataX.append(fp)
    return np.array(dataX)

#========================================================================================#
# Extract Mol2Vec features
#========================================================================================#
def Mol2Vec_extract(SMILES):
    if isinstance(SMILES, pd.DataFrame):
        df = SMILES
    else:
        df = pd.DataFrame(SMILES)
    df.columns = ['SMILES'] 
    model = word2vec.Word2Vec.load('./feature_extractor/mol2vec/models/model_300dim.pkl')
    df['ROMol'] = df.apply(lambda x: Chem.MolFromSmiles(x['SMILES'], 1), axis=1)
    df['sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['ROMol'], 1)), axis=1)
    df['mol2vec'] = [DfVec(x) for x in sentences2vec(df['sentence'], model, unseen='UNK')]
    X = np.array([x.vec for x in df['mol2vec']])
    return X    

#========================================================================================#
a = ['COc(cc1)ccc1C#N']
Mol2Vec_extract(a)
