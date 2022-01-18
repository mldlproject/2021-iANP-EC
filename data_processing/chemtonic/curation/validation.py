from rdkit import Chem
from .utils import *
import pandas as pd
import os

#==========================================================
# remove SMILES of mixtures
def rmMixtures(compounds, getMixtures=False, getMixturesIdx=False, printlogs=True):
    #------------------------
    if getMixturesIdx:
        if getMixtures == False:
            print("!!!ERROR: 'getMixturesIdx=True' argument goes with 'getMixtures=True'!!!")
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
    UniqueList, MixtureList, MixtureIdxList = [], [], []
    idx = 0
    for compound in compounds_:
        if compound.find('.') == -1:
            UniqueList.append(compound)
        else:
            if compound.find('.[') == -1:
                MixtureList.append(compound)
                MixtureIdxList.append(idx)
            else:
                if compound.count('.') == compound.count('.['):
                    UniqueList.append(compound)
                else:
                    MixtureList.append(compound)
                    MixtureIdxList.append(idx)
        idx +=1
    if printlogs:
        print("=======================================================")
        if Unverified_count > 0:
            print("Succeeded to verify {}/{} structures".format(len(compounds_), len(compounds)))
            print("Failed to verify {}/{} structures".format(Unverified_count, len(compounds)))
            print("Use function 'ultils.molValidate()' and set 'getFailedStruct=True' to get the list of unverified structures")
            if len(MixtureList) > 0:
                print("{}/{} structures are non-mixtures".format(len(UniqueList), len(compounds)))
                print("{}/{} structure(s) is/are mixture(s) and was/were removed".format(len(MixtureList), len(compounds)))   
            else:
                print("{}/{} structures are non-mixtures".format(len(UniqueList), len(compounds)))
        else:
            print("Succeeded to verify {}/{} structures".format(len(compounds), len(compounds)))
            if len(MixtureList) > 0:
                print("{}/{} structures are non-mixtures".format(len(UniqueList), len(compounds)))
                print("{}/{} structure(s) is/are mixture(s) and was/were removed".format(len(MixtureList), len(compounds)))
            else:
                print("{}/{} structures are non-mixtures".format(len(UniqueList), len(compounds)))
        print("=======================================================")
    if getMixtures:
        if getMixturesIdx:
            return MixtureList, MixtureIdxList
        else:
            return MixtureList   
    else:
        return UniqueList         
        
#==========================================================
# remove SMILES of inorganics
def rmInorganics(compounds, getInorganics=False, getInorganicsIdx=False, printlogs=True):
    #------------------------
    if getInorganicsIdx:
        if getInorganics == False:
            print("!!!ERROR: 'getInorganicsIdx=True' argument goes with 'getInorganics=True'!!!")
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
    excluded_list = ['Cr', 'Co', 'Cl', 'Cd', 'Cn', 'Cs', 'Ce', 'Cf', 'Cm']
    compounds_ = molStructVerify(compounds, printlogs=False)
    Unverified_count = len(molStructVerify(compounds, getFailedStruct=True, printlogs=False))
    OrganicsList, InorganicsList, InorganicsIdxList = [], [], []
    idx = 0
    for compound in compounds_:
        if compound.find('C') == -1 and compound.find('c') == -1:
            InorganicsList.append(compound)
            InorganicsIdxList.append(idx)
        else:
            if compound.count('C') > 1 or compound.count('c') > 1:
                OrganicsList.append(compound)
            else:
                result = map(lambda c: compound.find(c), excluded_list)
                if sum(list(result)) > len(excluded_list)/(-1):
                    InorganicsList.append(compound)
                    InorganicsIdxList.append(idx)
                else:
                    OrganicsList.append(compound)
        idx +=1
    if printlogs:
        print("=======================================================")
        if Unverified_count > 0:
            print("Succeeded to verify {}/{} structures".format(len(compounds_), len(compounds)))
            print("Failed to verify {}/{} structures".format(Unverified_count, len(compounds)))
            print("Use function 'ultils.molValidate()' and set 'getFailedStruct=True' to get the list of unverified structures")
            if len(InorganicsList) > 0:
                print("{}/{} structures are organics".format(len(OrganicsList), len(compounds)))
                print("{}/{} structure(s) is/are inorganic(s) and was/were removed".format(len(InorganicsList), len(compounds)))
            else:
                print("{}/{} structures are organics".format(len(OrganicsList), len(compounds)))
        else:
            print("Succeeded to verify {}/{} structures".format(len(compounds_), len(compounds)))
            if len(InorganicsList) > 0:
                print("{}/{} structures are organics".format(len(OrganicsList), len(compounds)))
                print("{}/{} structure(s) is/are inorganic(s) and was/were removed".format(len(InorganicsList), len(compounds)))
            else:
                print("{}/{} structures are organics".format(len(OrganicsList), len(compounds)))
        print("=======================================================")
    if getInorganics:
        if getInorganicsIdx:
            return InorganicsList, InorganicsIdxList
        else:
            return InorganicsList
    else:
        return OrganicsList        

#==========================================================
# remove SMILES of organometallics
def rmOrganometallics(compounds, getOrganometallics=False, getOrganometallicsIdx=False, printlogs=True):
    #------------------------
    if getOrganometallicsIdx:
        if getOrganometallics == False:
            print("!!!ERROR: 'getOrganometallicsIdx=True' argument goes with 'getOrganometallics=True'!!!")
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
    metals_list = open("./chemtonic/curation/dat/metals.txt").read().split(',')
    compounds_ = molStructVerify(compounds, printlogs=False)
    Unverified_count = len(molStructVerify(compounds, getFailedStruct=True, printlogs=False))
    OrganicsList, OrganometallicsList, OrganometallicsIdxList = [], [], []
    idx = 0
    for compound in compounds_:
        result = map(lambda c: compound.find(c), metals_list)
        if sum(list(result)) > sum(list(result))/(-1):
            if compound.find('.[') == -1:
                OrganometallicsList.append(compound)
                OrganometallicsIdxList.append(idx)
            else:
                OrganicsList.append(compound)
        else:
            OrganicsList.append(compound)
    if printlogs:
        print("=======================================================")
        if Unverified_count > 0:
            print("Succeeded to verify {}/{} structures".format(len(compounds_), len(compounds)))
            print("Failed to verify {}/{} structures".format(Unverified_count, len(compounds)))
            print("Use function 'ultils.molValidate()' and set 'getFailedStruct=True' to get the list of unverified structures")
            if len(OrganometallicsList) > 0:
                print("{}/{} structures are NOT organometallics".format(len(OrganicsList), len(compounds)))
                print("{}/{} structure(s) is/are organometallic(s) and was/were removed".format(len(OrganometallicsList), len(compounds)))
            else:
                print("{}/{} structures are NOT organometallics".format(len(OrganicsList), len(compounds)))
        else:
            print("Succeeded to verify {}/{} structures".format(len(compounds_), len(compounds)))
            if len(OrganometallicsList) > 0:
                print("{}/{} structures are NOT organometallics".format(len(OrganicsList), len(compounds)))
                print("{}/{} structure(s) is/are organometallic(s) and was/were removed".format(len(OrganometallicsList), len(compounds)))
            else:
                print("{}/{} structures are NOT organometallics".format(len(OrganicsList), len(compounds)))
        print("=======================================================")
    if getOrganometallics:
        if getOrganometallicsIdx:
            return OrganometallicsList, OrganometallicsIdxList
        else:
            OrganometallicsList
    else:
        return OrganicsList   

#==========================================================
# Complete validation
def validateComplete(compounds, getInvalidStruct=False, removeDuplicates=False, getDuplicatedIdx=False, exportCSV=False, outputPath=None, printlogs=True):
    #------------------------
    if getInvalidStruct:
        if removeDuplicates:
            print("!!!ERROR: 'removeDuplicates=True' argument goes with 'getInvalidStruct=False' only !!!")
            return None
        if getDuplicatedIdx:
            print("!!!ERROR: 'getDuplicatedIdx=True' argument goes with 'getInvalidStruct=False' only !!!")  
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
    compounds_r2     = rmMixtures(compounds_r1, printlogs=False) #test
    MixturesList, MixturesIdxList = rmMixtures(compounds_r1, getMixtures=True, getMixturesIdx=True, printlogs=False)
    Mixtures_count   = len(MixturesList)
    #------------------------
    compounds_r3     = rmInorganics(compounds_r1, printlogs=False) #test
    InorganicsList, InorganicsIdxList   = rmInorganics(compounds_r1, getInorganics=True, getInorganicsIdx=True, printlogs=False)
    Inorganics_count = len(InorganicsList)
    #------------------------
    compounds_r4          = rmOrganometallics(compounds_r1, printlogs=False) #test
    OrganometallicsList, OrganometallicsIdxList = rmOrganometallics(compounds_r1, getOrganometallics=True, getOrganometallicsIdx=True, printlogs=False)
    Organometallics_count = len(OrganometallicsList)
    #------------------------
    if printlogs:
        if Unverified_count > 0:
            print("=======================================================")
            print("Succeeded to verify {}/{} structures".format(len(compounds_r1), len(compounds)))
            print("Failed to verify {}/{}  structure(s) and it/they was/were removed \n".format(Unverified_count), len(compounds))
        else:
            print("=======================================================")
            print("Succeeded to validate {}/{} structures \n".format(len(compounds_r1), len(compounds)))   
        if Mixtures_count > 0:
            print("=======================================================")
            print("{}/{} structures are non-mixtures".format(len(compounds_r2), len(compounds)))
            print("{}/{} structure(s) is/are mixture(s) and was/were removed \n".format(Mixtures_count, len(compounds)))
        else:
            print("=======================================================")
            print("{}/{} structures are non-mixtures \n".format(len(compounds_r2), len(compounds)))
        if Inorganics_count > 0:
            print("=======================================================")
            print("{}/{} structures are organics".format(len(compounds_r3), len(compounds)))
            print("{}/{} structure(s) í/are inorganic(s) and was/were removed \n".format(Inorganics_count, len(compounds)))
        else:
            print("=======================================================")
            print("{}/{} structures are organics \n".format(len(compounds_r3), len(compounds)))
        if Organometallics_count > 0:
            print("=======================================================")
            print("{}/{} structures are NOT organometallics".format(len(compounds_r4), len(compounds)))
            print("{}/{} structure(s) is/are organometallic(s) and was/were removed \n".format(Organometallics_count, len(compounds)))
        else:
            print("=======================================================")
            print("{}/{} structures are NOT organometallics \n".format(len(compounds_r4), len(compounds)))
    #------------------------
    ValidatedList    = intersectionList(compounds_r1, intersectionList(compounds_r2, intersectionList(compounds_r3, compounds_r4)))
    UnvalidatedList  = UnverifiedList + MixturesList + InorganicsList + OrganometallicsList
    UnvalidatedLabel = len(UnverifiedList)*["UnverifiedStruct"] + len(MixturesList)*["Mixture"] + len(InorganicsList)*["Inorganic"] + len(OrganometallicsList)*["Organometallic"]
    FunctionLabel    = len(UnverifiedList)*["molStructVerify()"] + len(MixturesList)*["rmMixtures()"] + len(InorganicsList)*["rmInorganics()"] + len(OrganometallicsList)*["rmOrganometallics()"]
    IdxLabel         = UnverifiedIdxList + MixturesIdxList + InorganicsIdxList + OrganometallicsIdxList
    df1 = pd.DataFrame(zip(UnvalidatedList, UnvalidatedLabel, FunctionLabel, IdxLabel), columns=['SMILES', 'errorTag', 'fromFunction', 'idx'])
    if printlogs:
        print("=======================================================")
        print("SUMMARY:")
        print("{}/{} structures were successfully validated".format(len(ValidatedList), len(compounds)))   
        print("{}/{} structure(s) were/was unsuccessfully validated and need to be rechecked".format(len(UnvalidatedList), len(compounds)))
        print("-----------------------------------------------------------------------------")
        if len(UnvalidatedList) > 0:
            if getInvalidStruct == False:
                print("set 'getInvalidStruct=True' to get the list of all unvalidated structures \n")  
    #------------------------
    if getInvalidStruct:
        if exportCSV:
            filePath = outputPath + "UnvalidatedList.csv"
            if os.path.isdir(outputPath):
                df1.to_csv(filePath, index=False)
            else:
                os.makedirs(outputPath)
                df1.to_csv(filePath, index=False)
        else:
            return df1
    else:
        if removeDuplicates:
            DeduplicatedValidatedList = molDeduplicate(ValidatedList, printlogs=False)
            _, DuplicatedValidatedIdxList = molDeduplicate(ValidatedList, getDuplicates=True, getDuplicatesIdx=True, printlogs=False) #test
            if len(DeduplicatedValidatedList) == len(ValidatedList):
                if printlogs:
                    print("=======================================================")
                    print("No duplicate was found (in {} verified structures)".format(len(ValidatedList)))
                    print("-------------------------------------------------------")
                if getDuplicatedIdx:
                    df0 = pd.DataFrame(DuplicatedValidatedIdxList)
                    df0.columns = ['idx', 'matchedIdx']
                    df0['fromFunction'] = 'validateComplete()'
                    if exportCSV:
                        filePath = outputPath + "DuplicatedValidatedIdxList.csv"
                        if os.path.isdir(outputPath):
                            df0.to_csv(filePath, index=False)
                        else:
                            os.makedirs(outputPath)
                        df0.to_csv(filePath, index=False)
                    else:
                        return df0
                else: 
                    df0 = pd.DataFrame(DeduplicatedValidatedList)
                    df0.columns = ['SMILES']
                    if exportCSV:
                        filePath = outputPath + "NoDuplicatedValidatedIdxList.csv"
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
                    print("There are {} unique structures filtered from {} initial validated structures".format(len(DeduplicatedValidatedList), len(ValidatedList)))
                    print("-----------------------------------------------------------------------------")
                    print("To get detailed information, please follow steps below:")
                    print("(1) Rerun validateComplete() with setting 'removeDuplicates=False' to get the list of all validated structures")
                    print("(2) Run ultils.molDeduplicate() with setting 'getDuplicates=True'to get the list of duplicated structures \n")
                    print("--OR--")
                    print("Rerun validateComplete() with setting 'getDuplicates=True', 'exportCSV'=True, and 'outputPath=<Directory>' to export a csv file  containing the list of duplicated structures \n")
                    print("--OR--")
                    print("Run ultils.molDeduplicate() with settings 'getDuplicates=True' and 'getDuplicatesIndex=True' to get the list of duplicated structures with detailed indices")
                if getDuplicatedIdx:
                    df2 = pd.DataFrame(DuplicatedValidatedIdxList)
                    df2.columns = ['idx', 'matchedIdx']
                    df2['fromFunction'] = 'validateComplete()'
                    if exportCSV:
                        filePath = outputPath + "DuplicatedValidatedIdxList.csv"
                        if os.path.isdir(outputPath):
                            df2.to_csv(filePath, index=False)
                        else:
                            os.makedirs(outputPath)
                        df2.to_csv(filePath, index=False)
                    else:
                        return df2
                else:
                    df2 = pd.DataFrame(DeduplicatedValidatedList)
                    df2.columns = ['SMILES']
                    if exportCSV:
                        filePath = outputPath + "DeduplicatedValidatedList.csv"
                        if os.path.isdir(outputPath):
                            df2.to_csv(filePath, index=False)
                        else:
                            os.makedirs(outputPath)
                        df2.to_csv(filePath, index=False)
                    else:
                        return df2
        else:
            df3 = pd.DataFrame(ValidatedList)
            df3.columns = ['SMILES']
            if exportCSV:
                filePath = outputPath + "DuplicatedValidatedList.csv"
                if os.path.isdir(outputPath):
                    df3.to_csv(filePath, index=False)
                else:
                    os.makedirs(outputPath)
                    df3.to_csv(filePath, index=False)
            else:
                return df3            

        
            
            