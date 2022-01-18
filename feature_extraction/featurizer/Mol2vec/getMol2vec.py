import os
import re
import pandas as pd
import numpy as np
from rdkit import Chem
from .mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

def extract_Mol2Vec(compounds, getFailedSMILES=False, exportCSV=False, outputPath=None, tag=None):
    #----------------------------------
    if exportCSV:
        if outputPath == None:
            print("!!!ERROR 'exportCSV=True' needs 'outputPath=<Directory>' to be filled !!!")
            return None  
    #----------------------------------
    if isinstance(compounds, pd.core.series.Series):
        compounds = compounds.tolist()
    if isinstance(compounds, pd.core.frame.DataFrame):
        compounds = compounds.iloc[:,0].tolist()
    if isinstance(compounds, str):
        compounds = [compounds]
    if isinstance(compounds, list):
        compounds = compounds
    #----------------------------------
    colnames = ('dim-'+ str(i) for i in range(300))
    colnames = list(colnames)
    #----------------------------------
    df, error = [], []
    for compound in compounds:
        mol = Chem.MolFromSmiles(compound)
        if mol != None:
            df.append(compound)
        else:
            error.append(compound)
    df = pd.DataFrame(df)
    df.columns = ['SMILES']
    if len(error) > 0:
        error = pd.DataFrame(error)
        error.columns = ['Failed_SMILES']
    #----------------------------------
    model = word2vec.Word2Vec.load('./featurizer/Mol2vec/mol2vec/models/model_300dim.pkl')
    df['ROMol'] = df.apply(lambda x: Chem.MolFromSmiles(x['SMILES'], 1), axis=1)
    df['sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['ROMol'], 1)), axis=1)
    df['mol2vec'] = [DfVec(x) for x in sentences2vec(df['sentence'], model, unseen='UNK')]
    X = np.array([x.vec for x in df['mol2vec']])
    #----------------------------------
    if exportCSV:
        if tag == None:
            tag_ = '' 
        else:
            tag_ = tag
        filePath = outputPath + "Mol2vec" + tag_ + ".csv"
        #----------------------------------
        Mol2vec_df = pd.DataFrame(X)
        Mol2vec_df.columns = colnames
        #----------------------------------
        if os.path.isdir(outputPath):
            Mol2vec_df.to_csv(filePath, index=False)
        else:
            os.makedirs(outputPath)
            Mol2vec_df.to_csv(filePath, index=False)
    else:
        if getFailedSMILES:
            if len(error) > 0:
                return error
            else:
                print("No faied SMILES found")
                return None
        else:
            return X    

#========================================================================================#
# a = ['COc(cc1)ccc1C#N']
# extract_Mol2Vec(a)
