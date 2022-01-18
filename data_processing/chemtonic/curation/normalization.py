from rdkit import Chem
from .utils import *
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
import os
import pandas as pd

#==========================================================
# process SMILES of tautomers
def normTautomers(compounds, getTautomers=False, getTautomersIdx=False, deTautomerize=False, printlogs=True):
    #------------------------
    if getTautomersIdx:
        if getTautomers == False:
            print("!!!ERROR: 'getTautomersIdx=True' argument goes with 'getTautomers=True'!!!")
            return None
    #------------------------
    if isinstance(compounds, pd.core.series.Series):
        compounds = compounds.tolist()
    if isinstance(compounds, pd.core.frame.DataFrame):
        compounds = compounds.iloc[:,0].tolist()
    if isinstance(compounds, str):
        compounds = [compounds]
    if isinstance(compounds, list):
        compounds = compounds
    #------------------------
    compounds_ = molStructVerify(compounds, printlogs=False)
    Unverified_count = len(molStructVerify(compounds, getFailedStruct=True, printlogs=False))
    NonTautomersList, TautomersList, TautomersIdxList = [], [], []
    enumerator = rdMolStandardize.TautomerEnumerator()
    Tautomer_count = 0
    idx = 0
    for compound in compounds_:
        Premol = Chem.MolFromSmiles(compound)
        canonicalized_mol = enumerator.Canonicalize(Premol)
        canonicalized_SMILES = Chem.MolToSmiles(canonicalized_mol)
        if canonicalized_SMILES == compound:
            NonTautomersList.append(canonicalized_SMILES)
        else:
            Tautomer_count +=1
            if deTautomerize:
                NonTautomersList.append(canonicalized_SMILES)
                TautomersList.append(compound)
                TautomersIdxList.append(idx)
            else:
                TautomersList.append(compound)
                TautomersIdxList.append(idx)
        idx += 1
    if printlogs:
        print("=======================================================")
        if Unverified_count > 0:
            print("Succeeded to verify {}/{} structures".format(len(compounds_), len(compounds)))
            print("Failed to verify {}/{} structures".format(Unverified_count, len(compounds)))
            print("Use function 'utils.molValidate' and set 'getFailedStruct=True' to get the list of unverified structures")
            if Tautomer_count > 0:
                if deTautomerize:
                    print("{}/{} structures are NOT tautomers".format(len(NonTautomersList)-Tautomer_count, len(compounds)))
                    print("{}/{} structure(s) is/are tautomer(s) BUT was/were detautomerized".format(Tautomer_count, len(compounds)))
                    print("=======================================================")
                    print("!!!!!Notice: Detautomerizing has been applied!!!!!")
                else:
                    print("{}/{} structures are NOT tautomers".format(len(NonTautomersList), len(compounds)))
                    print("{}/{} structure(s) is/are tautomer(s) BUT was/were NOT detautomerized".format(Tautomer_count, len(compounds)))
            else:
                print("{}/{} structures are NOT tautomers".format(len(NonTautomersList), len(compounds)))   
        else:
            print("Succeeded to verify {}/{} structures".format(len(compounds_), len(compounds)))
            if Tautomer_count > 0:
                if deTautomerize:
                    print("{}/{} structures are NOT tautomers".format(len(NonTautomersList)-Tautomer_count, len(compounds)))
                    print("{}/{} structure(s) í/are tautomer(s) BUT was/were detautomerized".format(Tautomer_count, len(compounds)))
                    print("=======================================================")
                    print("!!!!!Notice: Detautomerizing has been applied!!!!!")
                else:
                    print("{}/{} structures are NOT tautomers".format(len(NonTautomersList), len(compounds)))
                    print("{}/{} structure(s) is/are tautomer(s) BUT was/were NOT detautomerized".format(Tautomer_count, len(compounds)))
            else:
                print("{}/{} structures are NOT tautomers".format(len(NonTautomersList), len(compounds)))       
        print("=======================================================")
    if getTautomers:  
        if getTautomersIdx:
            return TautomersList, TautomersIdxList
        else:
            return TautomersList
    else:
        return NonTautomersList     

#==========================================================
# process SMILES of stereoisomers
def normStereoisomers(compounds, getStereoisomers=False, getStereoisomersIdx=False, deSterioisomerize=False, printlogs=True):
    #------------------------
    if getStereoisomersIdx:
        if getStereoisomers == False:
            print("!!!ERROR: 'getStereoisomersIdx=True' argument goes with 'getStereoisomers=True'!!!")
            return None
    #------------------------
    if isinstance(compounds, pd.core.series.Series):
        compounds = compounds.tolist()
    if isinstance(compounds, pd.core.frame.DataFrame):
        compounds = compounds.iloc[:,0].tolist()
    if isinstance(compounds, str):
        compounds = [compounds]
    if isinstance(compounds, list):
        compounds = compounds
    #------------------------
    compounds_ = molStructVerify(compounds, printlogs=False)
    Unverified_count = len(molStructVerify(compounds, getFailedStruct=True, printlogs=False))
    NonStereoisomersList, StereoisomersList, StereoisomersIdxList = [], [], []
    Stereoisomer_count = 0
    idx = 0
    for compound in compounds_:
        if compound.find("@") == -1 and compound.find("/") == -1 and compound.find("\\") == -1:
            NonStereoisomersList.append(compound)
        else:
            Stereoisomer_count +=1
            if deSterioisomerize:
                NonStereoisomersList.append(compound.replace("@", "").replace("/","").replace("\\",""))
                StereoisomersList.append(compound)
                StereoisomersIdxList.append(idx)
            else:
                StereoisomersList.append(compound)
                StereoisomersIdxList.append(idx)               
        idx += 1
    if printlogs:
        print("=======================================================")
        if Unverified_count > 0:
            print("Succeeded to verify {}/{} structures".format(len(compounds_), len(compounds)))
            print("Failed to verify {}/{} structures".format(Unverified_count, len(compounds)))
            print("Use function 'utils.molValidate' and set 'getFailedStruct=True' to get the list of unverified structures")
            if Stereoisomer_count > 0:
                if deSterioisomerize:
                    print("{}/{} structures are NOT stereoisomers".format(len(NonStereoisomersList)-Stereoisomer_count, len(compounds)))
                    print("{}/{} structure(s) is/are stereoisomer(s) BUT was/were destereoisomerized".format(Stereoisomer_count, len(compounds)))
                    print("=======================================================")
                    print("!!!!!Notice: Destereoisomerization has been applied!!!!!")
                else:
                    print("{}/{} structures are NOT stereoisomers".format(len(NonStereoisomersList), len(compounds)))
                    print("{}/{} structure(s) is/are stereoisomer(s) BUT was/were NOT destereoisomerized".format(Stereoisomer_count, len(compounds)))
            else:
                print("{}/{} structures are NOT stereoisomers".format(len(NonStereoisomersList), len(compounds)))                
        else:
            print("Succeeded to verify {}/{} structures".format(len(compounds_), len(compounds)))
            if Stereoisomer_count > 0:
                if deSterioisomerize:
                    print("{}/{} structures are NOT stereoisomers".format(len(NonStereoisomersList)-Stereoisomer_count, len(compounds)))
                    print("{}/{} structure(s) is/are stereoisomer(s) BUT was/were destereoisomerized".format(Stereoisomer_count, len(compounds)))
                    print("=======================================================")
                    print("!!!!!Notice: Destereoisomerization has been applied!!!!!")
                else:
                    print("{}/{} structures are NOT stereoisomers".format(len(NonStereoisomersList), len(compounds)))
                    print("{}/{} structure(s) is/are stereoisomer(s) BUT was/were NOT destereoisomerized".format(Stereoisomer_count, len(compounds)))
            else:
                print("{}/{} structures are NOT stereoisomers".format(len(NonStereoisomersList), len(compounds)))
        print("=======================================================")
    if getStereoisomers:  
        if getStereoisomersIdx:
            return StereoisomersList, StereoisomersIdxList
        else:
            return StereoisomersList
    else:
        return NonStereoisomersList     

#==========================================================         
# Complete normalization
def normalizeComplete(compounds, getUnnormalizedStruct=False, deTautomerize=True, deSterioisomerize=True, removeDuplicates=False, getDuplicatedIdx=False, exportCSV=False, outputPath=None, printlogs=True):
    #------------------------
    if getUnnormalizedStruct:
        if removeDuplicates:
            print("!!!ERROR: 'removeDuplicates=True' argument goes with 'getUnnormalizedStruct=False' only !!!")
            return None
        if getDuplicatedIdx:
            print("!!!ERROR: 'getDuplicatedIdx=True' argument goes with 'getUnnormalizedStruct=False' only !!!")  
    if getDuplicatedIdx:
        if removeDuplicates == False:
            print("!!!ERROR: 'getDuplicatedIdx=True' argument goes with 'removeDuplicates=True'!!!")
            return None  
    if exportCSV:
        if outputPath == None:
            print("!!!ERROR 'exportCSV=True' needs 'outputPath=<Directory>' to be filled !!!")
            return None  
    #------------------------
    if isinstance(compounds, pd.core.series.Series):
        compounds = compounds.tolist()
    if isinstance(compounds, pd.core.frame.DataFrame):
        compounds = compounds.iloc[:,0].tolist()
    if isinstance(compounds, str):
        compounds = [compounds]
    if isinstance(compounds, list):
        compounds = compounds
    #------------------------
    compounds_r1     = molStructVerify(compounds, printlogs=False)
    UnverifiedList, UnverifiedIdxList = molStructVerify(compounds, getFailedStruct=True, getFailedStructIdx=True, printlogs=False)
    Unverified_count = len(UnverifiedList)
    #------------------------
    if deSterioisomerize:
        compounds_r2 = normStereoisomers(compounds_r1, deSterioisomerize=True, printlogs=False)
    else:
        compounds_r2 = normStereoisomers(compounds_r1, deSterioisomerize=False, printlogs=False)
    StereoisomersList, StereoisomersIdxList = normStereoisomers(compounds_r1, getStereoisomers=True, getStereoisomersIdx=True, printlogs=False)
    Stereoisomers_count = len(StereoisomersList)
    #------------------------
    if deTautomerize:
        compounds_r3 = normTautomers(compounds_r2, deTautomerize=True, printlogs=False)
    else:
        compounds_r3 = normTautomers(compounds_r2, deTautomerize=False, printlogs=False)
    TautomersList, TautomersIdxList = normTautomers(compounds_r1, getTautomers=True, getTautomersIdx=True, printlogs=False)
    Tautomers_count  = len(TautomersList)
    #------------------------
    if printlogs:
        if Unverified_count > 0:
            print("=======================================================")
            print("Succeeded to verify {}/{} structures".format(len(compounds_r1), len(compounds)))
            print("Failed to verify {} structures \n".format(Unverified_count))
        else:
            print("=======================================================")
            print("Succeeded to validate {}/{} structures \n".format(len(compounds_r1), len(compounds))) 
        if Stereoisomers_count > 0:
            if deSterioisomerize:
                print("=======================================================")
                print("{}/{} structures are NOT stereoisomers".format(len(compounds_r2)-Stereoisomers_count, len(compounds)))
                print("{}/{} structure(s) is/are stereoisomer(s) BUT was/were destereoisomerized \n".format(Stereoisomers_count, len(compounds)))
            else:
                print("=======================================================")
                print("{}/{} structures are NOT stereoisomers".format(len(compounds_r2), len(compounds)))
                print("{}/{} structure(s) is/are stereoisomer(s) BUT was/were NOT destereoisomerized \n".format(Stereoisomers_count, len(compounds)))
        else:
            print("=======================================================")
            print("{}/{} structures are NOT stereoisomers".format(len(compounds_r2), len(compounds)))
        if Tautomers_count > 0:    
            if deTautomerize:
                print("=======================================================")
                compounds_r3_ = normTautomers(compounds_r2, deTautomerize=True, printlogs=False)
                print("{}/{} structures are NOT tautomers".format(len(compounds_r3_)-Tautomers_count, len(compounds)))
                print("{}/{} structure(s) is/are tautomer(s) BUT was/were detautomerized \n".format(Tautomers_count, len(compounds)))
            else:
                print("=======================================================")
                compounds_r3_ = normTautomers(compounds_r2, deTautomerize=False, printlogs=False)
                print("{}/{} structures are NOT tautomers".format(len(compounds_r3_), len(compounds)))
                print("{}/{} structure(s) is/are tautomer(s) but was/were NOT detautomerized \n".format(Tautomers_count, len(compounds)))              
        else:
            print("=======================================================")
            print("{}/{} structures are NOT tautomers \n".format(len(compounds_r2), len(compounds)))        
    #------------------------
    NormalizedList    = compounds_r3
    UnNormalizedList  = UnverifiedList + TautomersList + StereoisomersList
    UnNormalizedLabel = len(UnverifiedList)*["UnverifiedStruct"] + len(TautomersList)*["Tautomer"] + len(StereoisomersList)*["Stereoisomer"]
    FunctionLabel     = len(UnverifiedList)*["molStructVerify()"] + len(TautomersList)*["normTautomers()"] + len(StereoisomersList)*["normStereoisomers()"]
    IdxLabel         = UnverifiedIdxList + TautomersIdxList + StereoisomersIdxList 
    df1 = pd.DataFrame(zip(UnNormalizedList, UnNormalizedLabel, FunctionLabel, IdxLabel), columns=['SMILES', 'errorTag', 'fromFunction', 'idx'])
    #------------------------
    if printlogs:
        print("=======================================================")
        print("SUMMARY:")
        if len(UnverifiedList) > 0:
            print("{}/{} structures were successfully verfied".format(len(compounds_r1), len(compounds)))   
            print("{}/{} structure(s) was/were unsuccessfully verfied and need to be rechecked".format(len(UnverifiedList), len(compounds)))
        else:
            print("{}/{} structure were successfully verfied".format(len(compounds_r1), len(compounds)))    
        if len(UnNormalizedList) > 0:
            print("{}/{} structures were successfully normalized".format(len(NormalizedList), len(compounds)))
            if len(compounds_r1) > len(NormalizedList):    
                print("{}/{} structure(s) was/were unsuccessfully normalized and need to be rechecked".format(len(compounds_r1)-len(NormalizedList), len(compounds)))
            print("=======================================================")
        else:
            print("{}/{} structures were successfully normalized".format(len(NormalizedList), len(compounds)))   
            print("-------------------------------------------------------")
        if len(UnNormalizedList) > 0:
            if getUnnormalizedStruct == False:
                print("set 'getUnnormalizedStruct=True' to get the list of all unnormalized structures. \n")  
    #------------------------
    if getUnnormalizedStruct:
        if exportCSV:
            filePath = outputPath + "UnnormalizedList.csv"
            if os.path.isdir(outputPath):
                df1.to_csv(filePath, index=False)
            else:
                os.makedirs(outputPath)
                df1.to_csv(filePath, index=False)
        else:
            return df1
    else:
        if removeDuplicates:
            DeduplicatedNormalizedList = molDeduplicate(NormalizedList, printlogs=False)
            _, DuplicatedNormalizedIdxList = molDeduplicate(NormalizedList, getDuplicates=True, getDuplicatesIdx=True, printlogs=False)
            if len(DeduplicatedNormalizedList) == len(NormalizedList):
                if printlogs:
                    print("No duplicate was found (in {} normalized structures)".format(len(NormalizedList)))
                if getDuplicatedIdx:
                    df0 = pd.DataFrame(DuplicatedNormalizedIdxList)
                    df0.columns = ['idx', 'matchedIdx']
                    df0['fromFunction'] = 'normalizeComplete()'
                    if exportCSV:
                        filePath = outputPath + "DuplicatedNormalizedIdxList.csv"
                        if os.path.isdir(outputPath):
                            df0.to_csv(filePath, index=False)
                        else:
                            os.makedirs(outputPath)
                        df0.to_csv(filePath, index=False)
                    else:
                        return df0
                else: 
                    df0 = pd.DataFrame(DeduplicatedNormalizedList)
                    df0.columns = ['SMILES']
                    if exportCSV:
                        filePath = outputPath + "NoDuplicatedNormalizedIdxList.csv"
                        if os.path.isdir(outputPath):
                            df0.to_csv(filePath, index=False)
                        else:
                            os.makedirs(outputPath)
                        df0.to_csv(filePath, index=False)
                    else:
                        return df0                   
            else:
                if printlogs:
                    print("=============================================================================")
                    print("There are {} unique structures filtered from {} initial normalized structures".format(len(DeduplicatedNormalizedList), len(NormalizedList)))
                    print("=============================================================================")
                    print("To get detailed information, please follow steps below:")
                    print("(1) Rerun normalizeComplete() with setting 'removeDuplicates=False' to get the list of all normalized structures")
                    print("(2) Run ultils.molDeduplicate() with setting 'getDuplicates=True'to get the list of duplicated structures \n")
                    print("--OR--")
                    print("Rerun normalizeComplete() with setting 'getDuplicates=True', 'exportCSV'=True, and 'outputPath=<Directory>' to export a csv file  containing the list of duplicated structures \n")
                    print("--OR--")
                    print("Run ultils.molDeduplicate() with settings 'getDuplicates=True' and 'getDuplicatesIndex=True' to get the list of duplicated structures with detailed indices")
                if getDuplicatedIdx:
                    df2 = pd.DataFrame(DuplicatedNormalizedIdxList)
                    df2.columns = ['idx', 'matchedIdx']
                    df2['fromFunction'] = 'normalizeComplete()'
                    if exportCSV:
                        filePath = outputPath + "DuplicatedNormalizedIdxList.csv"
                        if os.path.isdir(outputPath):
                            df2.to_csv(filePath, index=False)
                        else:
                            os.makedirs(outputPath)
                        df2.to_csv(filePath, index=False)
                    else:
                        return df2    
                else:
                    df2 = pd.DataFrame(DeduplicatedNormalizedList)
                    df2.columns = ['SMILES']
                    if exportCSV:
                        filePath = outputPath + "DeduplicatedNormalizedList.csv"
                        if os.path.isdir(outputPath):
                            df2.to_csv(filePath, index=False)
                        else:
                            os.makedirs(outputPath)
                        df2.to_csv(filePath, index=False)
                    else:      
                        return df2                 
        else:
            df3 = pd.DataFrame(NormalizedList)
            df3.columns = ['SMILES']
            if exportCSV:
                filePath = outputPath + "DuplicatedNormalizedList.csv"
                if os.path.isdir(outputPath):
                    df3.to_csv(filePath, index=False)
                else:
                    os.makedirs(outputPath)
                    df3.to_csv(filePath, index=False)
            else:
                return df3       