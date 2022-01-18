import os
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem


def extract_MorganFP(compounds, exportCSV=False, outputPath=None, tag=None, radius=2, bit_type=1024):
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
    colnames = ('bit-'+ str(i) for i in range(bit_type))
    colnames = list(colnames)
    MorganFP_df = pd.DataFrame(columns=colnames)
    #----------------------------------
    compounds_df = pd.DataFrame(compounds)
    compounds_df.columns = ['SMILES']
    #----------------------------------
    for compound in compounds:
        mol = Chem.MolFromSmiles(compound)
        if mol != None:
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits = bit_type))
            fp_df = pd.DataFrame(fp).T
            fp_df.columns = colnames
        else:
            fp_df = pd.DataFrame([['na']*bit_type])
            fp_df.columns = colnames
        #----------------------------------
        MorganFP_df = pd.concat([MorganFP_df, fp_df])
    #----------------------------------
    MorganFP_df = MorganFP_df.reset_index(drop=True)
    MorganFP_df = pd.concat([compounds_df, MorganFP_df], axis=1)
    #----------------------------------
    if exportCSV:
        if tag == None:
            tag_ = '' 
        else:
            tag_ = tag
        filePath = outputPath + "MorganFP" + tag_ + ".csv"
        if os.path.isdir(outputPath):
            MorganFP_df.to_csv(filePath, index=False)
        else:
            os.makedirs(outputPath)
            MorganFP_df.to_csv(filePath, index=False)
    else:
         return MorganFP_df
