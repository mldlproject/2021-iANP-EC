from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover
from .utils import *
import pandas as pd
import os

#==========================================================
# remove SMILES of salts
def clSalts(compounds, getSalts=False, getSaltsIdx=False, deSalt=False, printlogs=True):
    #------------------------
    if getSaltsIdx:
        if getSalts == False:
            print("!!!ERROR: 'getSaltsIdx=True' argument goes with 'getSalts=True'!!!")
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
    #------------------------
    remover = SaltRemover()
    NonSaltsList, SaltsList, SaltsIdxList = [], [], []
    saltCount = 0
    idx = 0
    for compound in compounds_:
        PreMol  = Chem.MolFromSmiles(compound)
        res     = remover.StripMol(PreMol)
        PostSMILES = Chem.MolToSmiles(res)
        if compound == PostSMILES:
            NonSaltsList.append(PostSMILES)
        else:
            saltCount += 1
            if deSalt:
                NonSaltsList.append(PostSMILES)
                SaltsList.append(compound)
                SaltsIdxList.append(idx)
            else:
                SaltsList.append(compound)
                SaltsIdxList.append(idx)
        idx += 1
    if printlogs:
        print("=======================================================")
        if Unverified_count > 0:
            print("Succeeded to verify {}/{} structures".format(len(compounds_), len(compounds)))
            print("Failed to verify {}/{} structures".format(Unverified_count, len(compounds)))
            print("Use function 'utils.molValidate()' and set 'getFailedStruct=True' to get the list of unverified structures")
            if saltCount > 0:
                if deSalt:
                    print("{}/{} structures are NOT salts".format(len(NonSaltsList)-saltCount, len(compounds)))
                    print("{}/{} structure(s) is/are salt(s) BUT was/were desalted".format(saltCount, len(compounds)))  
                    print("=======================================================")
                    print("!!!!!Notice: Desalting compound(s) is not recommended without reasonable purpose!!!!!")   
                else:
                    print("{}/{} structures are NOT salts".format(len(NonSaltsList), len(compounds)))
                    print("{}/{} structure(s) is/are salt(s) and was/were removed".format(saltCount, len(compounds)))
            else:
                print("{}/{} structures are NOT salts".format(len(NonSaltsList), len(compounds)))
        else:
            print("Succeeded to verify {}/{} structures".format(len(compounds_), len(compounds)))
            if saltCount > 0:
                if deSalt:
                    print("{}/{} structures are NOT salts".format(len(NonSaltsList)-saltCount, len(compounds)))
                    print("{}/{} structure(s) is/are salt(s) BUT was/were desalted".format(saltCount, len(compounds)))
                    print("=======================================================")
                    print("!!!!!Notice: Desalting compound(s) is not recommended without reasonable purpose!!!!!")   
                else:
                    print("{}/{} structures are NOT salts".format(len(NonSaltsList), len(compounds)))
                    print("{}/{} structure(s) is/are salt(s) and was/were removed".format(saltCount, len(compounds)))
            else:
                print("{}/{} structures are NOT salts".format(len(NonSaltsList), len(compounds)))
        print("=======================================================")
    #------------------------
    if getSalts:
        if getSaltsIdx:
            return SaltsList, SaltsIdxList
        else:
            return SaltsList
    else:
        return NonSaltsList                    
        
#==========================================================
# remove SMILES of charged compounds
def clCharges(compounds, getCharges=False, getChargesIdx=False, deCharges=False, printlogs=True):
    #------------------------
    if getChargesIdx:
        if getCharges == False:
            print("!!!ERROR: 'getChargesIdx=True' argument goes with 'getCharges=True'!!!")
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
    #------------------------
    NonChargesList, ChargesList, ChargesIdxList = [], [], []
    Charge_count = 0
    idx =0
    for compound in compounds_:
        PreMol  = Chem.MolFromSmiles(compound)
        res     = neutralize_atoms(PreMol)
        PostSMILES = Chem.MolToSmiles(res)
        if compound == PostSMILES:
            NonChargesList.append(PostSMILES)
        else:
            Charge_count +=1
            if deCharges:
                NonChargesList.append(PostSMILES)
                ChargesList.append(compound)
                ChargesIdxList.append(idx)
            else:
                NonChargesList.append(compound)
                ChargesList.append(compound)
                ChargesIdxList.append(idx)
        idx += 1
    if printlogs:
        print("=======================================================")
        if Unverified_count > 0:
            print("Succeeded to verify {}/{} structures".format(len(compounds_), len(compounds)))
            print("Failed to verify {}/{} structures".format(Unverified_count, len(compounds)))
            print("Use function 'molValidate' and set 'getFailedStruct=True' to get the list of unverified structures")
            if Charge_count > 0:
                if deCharges:
                    print("{}/{} structures are NOT charges".format(len(NonChargesList)-Charge_count, len(compounds)))
                    print("{}/{} structure(s) is/are charge(s) BUT was/were neutralized".format(Charge_count, len(compounds)))
                    print("=======================================================")
                    print("!!!!!Notice: Neutralizing charged compound(s) is not recommended without reasonable purpose!!!!!")   
                else:
                    print("{}/{} structures are NOT charges".format(len(NonChargesList)-Charge_count, len(compounds)))
                    print("{}/{} structure(s) is/are charge(s) BUT was/were NOT neutralized".format(Charge_count, len(compounds))) 
            else:
                print("{}/{} structures are NOT charges".format(len(NonChargesList), len(compounds)))    
        else:
            print("Succeeded to verify {}/{} structures".format(len(compounds_), len(compounds)))
            if Charge_count > 0:
                if deCharges:
                    print("{}/{} structures are NOT charges".format(len(NonChargesList)-Charge_count, len(compounds)))
                    print("{}/{} structure(s) is/are charge(s) BUT was/were neutralized".format(Charge_count, len(compounds)))
                    print("=======================================================")
                    print("!!!!!Notice: Neutralizing charged compound(s) is not recommended without reasonable purpose!!!!!")  
                else:
                    print("{}/{} structures are NOT charges".format(len(NonChargesList)-Charge_count, len(compounds)))
                    print("{}/{} structure(s) is/are charge(s) BUT was/were NOT neutralized".format(Charge_count, len(compounds)))
            else:
                print("{}/{} structures are NOT charges".format(len(NonChargesList), len(compounds)))             
        print("=======================================================")
    if getCharges:
        if getChargesIdx:
            return ChargesList, ChargesIdxList
        else:
            return ChargesList
    else:
        return NonChargesList   
    
#==========================================================
# Complete cleaning
def cleanComplete(compounds, getUncleanedStruct=False, deSalt=False, neutralize=False, removeDuplicates=False,  getDuplicatedIdx=False, exportCSV=False, outputPath=None, printlogs=True):
    #------------------------
    if getUncleanedStruct:
        if removeDuplicates:
            print("!!!ERROR: 'removeDuplicates=True' argument goes with 'getUncleanedStruct=False' only !!!")
            return None
        if getDuplicatedIdx:
            print("!!!ERROR: 'getDuplicatedStruct=True' argument goes with 'getUncleanedStruct=False' only !!!")
            return None
    if getDuplicatedIdx:
        if removeDuplicates == False:
            print("!!!ERROR: 'getDuplicatedStruct=True' argument goes with 'removeDuplicates=True'!!!")
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
    if deSalt:
        compounds_r2  = clSalts(compounds_r1, deSalt=True, printlogs=False) #test
        if neutralize:
            compounds_r3  = clCharges(compounds_r2, deCharges=True, printlogs=False) 
        else:
            compounds_r3  = clCharges(compounds_r2, deCharges=False, printlogs=False) 
    else:
        compounds_r2  = clSalts(compounds_r1, deSalt=False, printlogs=False) #test
        if neutralize:
            compounds_r3  = clCharges(compounds_r2, deCharges=True, printlogs=False)
        else:
            compounds_r3  = clCharges(compounds_r2, deCharges=False, printlogs=False)
    #------------------------
    SaltsList, SaltsIdxList  = clSalts(compounds_r1, getSalts=True, getSaltsIdx=True, printlogs=False) #test
    Salts_count   = len(SaltsList)
    #------------------------
    ChargesList, ChargesIdxList   = clCharges(compounds_r1, getCharges=True, getChargesIdx=True, printlogs=False) #test
    Charges_count = len(ChargesList)
    #------------------------
    if printlogs:
        if Unverified_count > 0:
            print("=======================================================")
            print("Succeeded to verify {}/{} structures".format(len(compounds_r1), len(compounds)))
            print("Failed to verify {} structures \n".format(Unverified_count))
        else:
            print("=======================================================")
            print("Succeeded to validate {}/{} structures \n".format(len(compounds_r1), len(compounds)))   
        if Salts_count > 0:
            if deSalt:
                print("=======================================================")
                print("{}/{} structures are NOT salts".format(len(compounds_r2)-Salts_count, len(compounds)))
                print("{}/{} structure(s) is/are salt(s) BUT was/were desalted \n".format(Salts_count, len(compounds)))
            else:
                print("=======================================================")
                print("{}/{} structures are NOT salts".format(len(compounds_r2), len(compounds)))
                print("{}/{} structure(s) is/are salt(s) and was/were removed \n".format(Salts_count, len(compounds)))
        else:
            print("=======================================================")
            print("{}/{} structures are NOT salts".format(len(compounds_r2), len(compounds)))
        if Charges_count > 0:
            if neutralize:   
                print("=======================================================")
                compounds_r3_ = clCharges(compounds_r1, deCharges=True, printlogs=False)
                print("{}/{} structures are NOT charges".format(len(compounds_r3_)-Charges_count, len(compounds)))
                print("{}/{} structure(s) is/are charge(s) BUT was/were neutralized \n".format(Charges_count, len(compounds)))
            else:
                print("=======================================================")
                compounds_r3_ = clCharges(compounds_r1, deCharges=False, printlogs=False)
                print("{}/{} structures are NOT charges".format(len(compounds_r3_)-Charges_count, len(compounds)))
                print("{}/{} structure(s) is/are charge(s) BUT was/were NOT neutralized \n".format(Charges_count, len(compounds)))
        else:
            print("=======================================================")
            print("{}/{} structures are NOT charges".format(len(compounds_r3), len(compounds)))
    #------------------------
    CleanedList    = compounds_r3
    UnCleanedList  = UnverifiedList + SaltsList + ChargesList
    UnCleanedLabel = len(UnverifiedList)*["UnverifiedStruct"] + len(SaltsList)*["Salt"] + len(ChargesList)*["Charge"] 
    FunctionLabel  = len(UnverifiedList)*["molStructVerify()"] + len(SaltsList)*["clSalts()"] + len(ChargesList)*["clCharges()"] 
    IdxLabel       = UnverifiedIdxList + SaltsIdxList + ChargesIdxList 
    df1 = pd.DataFrame(zip(UnCleanedList, UnCleanedLabel, FunctionLabel, IdxLabel), columns=['SMILES', 'errorTag', 'fromFunction', 'idx'])
    #------------------------
    if printlogs:
        print("=======================================================")
        print("SUMMARY:")
        if len(UnverifiedList) > 0:
            print("{}/{} structures were successfully verfied".format(len(compounds_r1), len(compounds)))   
            print("{}/{} structure(s) was/were unsuccessfully verfied and need to be rechecked".format(len(UnverifiedList), len(compounds)))
        else:
            print("{}/{} structures were successfully verfied".format(len(compounds_r1), len(compounds)))    
        if len(UnCleanedList) > 0:
            print("{}/{} structures were successfully cleaned".format(len(CleanedList), len(compounds)))   
            if len(compounds_r1) > len(CleanedList):
                print("{}/{} structure(s) was/were unsuccessfully cleaned and need to be rechecked".format(len(compounds_r1) - len(CleanedList), len(compounds)))
            if deSalt:
                print("!!!!!Notice: Desalting compound(s) is not recommended without reasonable purpose!!!!!")
            if neutralize:
                print("!!!!!Notice: Neutralizing charged structure(s) is not recommended without reasonable purpose!!!!!")
        else:
            print("{}/{} structures were successfully cleaned".format(len(CleanedList), len(compounds)))   
            if deSalt:
                print("!!!!!Notice: Desalting compound(s) is not recommended without reasonable purpose!!!!!")
            if neutralize:
                print("!!!!!Notice: Neutralizing charged structure(s) is not recommended without reasonable purpose!!!!!")
        print("-------------------------------------------------------")  
        if len(UnCleanedList) > 0:
            if getUncleanedStruct == False:
                print("set 'getUncleanedStruct=True' to get the list of all uncleaned structures. Neutralized charged structures will be included (if any) \n")  
    #------------------------
    if getUncleanedStruct:
        if exportCSV:
            filePath = outputPath + "UncleanedList.csv"
            if os.path.isdir(outputPath):
                df1.to_csv(filePath, index=False)
            else:
                os.makedirs(outputPath)
                df1.to_csv(filePath, index=False)
        else:
            return df1
    else:
        if removeDuplicates:
            DeduplicatedCleanedList = molDeduplicate(CleanedList, printlogs=False)
            _, DuplicatedCleanedIdxList = molDeduplicate(CleanedList, getDuplicates=True, getDuplicatesIdx=True, printlogs=False)
            if len(DeduplicatedCleanedList) == len(CleanedList):
                if printlogs:
                    print("No duplicate was found (in {} cleaned structures)".format(len(CleanedList)))
                if getDuplicatedIdx:
                    df0 = pd.DataFrame(DuplicatedCleanedIdxList)
                    df0.columns = ['idx', 'matchedIdx']
                    df0['fromFunction'] = 'cleanComplete()'
                    if exportCSV:
                        filePath = outputPath + "DuplicatedCleanedIdxList.csv"
                        if os.path.isdir(outputPath):
                            df0.to_csv(filePath, index=False)
                        else:
                            os.makedirs(outputPath)
                        df0.to_csv(filePath, index=False)
                    else:
                        return df0
                else:
                    df0 = pd.DataFrame(DeduplicatedCleanedList)
                    df0.columns = ['SMILES']
                    if exportCSV:
                        filePath = outputPath + "NoDuplicatedCleanedIdxList.csv"
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
                    print("There are {} unique structures filtered from {} initial cleaned structures".format(len(DeduplicatedCleanedList), len(CleanedList)))
                    print("-----------------------------------------------------------------------------")
                    print("To get detailed information, please follow steps below:")
                    print("(1) Rerun cleanComplete() with setting 'removeDuplicates=False' to get the list of all validated structures")
                    print("(2) Run ultils.molDeduplicate() with setting 'getDuplicates=True'to get the list of duplicated structures \n")
                    print("--OR--")
                    print("Rerun cleanComplete() with setting 'getDuplicates=True', 'exportCSV=True', and 'outputPath'=<Directory>' to export a csv file  containing the list of duplicated structures \n")
                    print("--OR--")
                    print("Run ultils.molDeduplicate() with settings 'getDuplicates=True' and 'getDuplicatesIndex=True' to get the list of duplicated structures with detailed indices")
                    print("--OR--")
                if getDuplicatedIdx:
                    df2 = pd.DataFrame(DuplicatedCleanedIdxList)
                    df2.columns = ['idx', 'matchedIdx']
                    df2['fromFunction'] = 'cleanComplete()'
                    if exportCSV:
                        filePath = outputPath + "DuplicatedCleanedIdxList.csv"
                        if os.path.isdir(outputPath):
                            df2.to_csv(filePath, index=False)
                        else:
                            os.makedirs(outputPath)
                        df2.to_csv(filePath, index=False)
                    else:
                        return df2
                else:
                    df2 = pd.DataFrame(DeduplicatedCleanedList)
                    df2.columns = ['SMILES']
                    if exportCSV:
                        filePath = outputPath + "DeduplicatedCleanedList.csv"
                        if os.path.isdir(outputPath):
                            df2.to_csv(filePath, index=False)
                        else:
                            os.makedirs(outputPath)
                        df2.to_csv(filePath, index=False)
                    else:
                        return df2
        else:
            df3 = pd.DataFrame(CleanedList)
            df3.columns = ['SMILES']
            if exportCSV:
                filePath = outputPath + "DuplicatedCleanedList.csv"
                if os.path.isdir(outputPath):
                    df3.to_csv(filePath, index=False)
                else:
                    os.makedirs(outputPath)
                    df3.to_csv(filePath, index=False)
            else:
                return df3            


