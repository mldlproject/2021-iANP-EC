from .validation import *
from .cleaning import *
from .normalization import *
from .utils import *
import pandas as pd
import os

#==========================================================        
# Complete refinement
def refineComplete(compounds, getUnrefinedStruct=False, deSalt=False, neutralize=False, deTautomerize=True, deSterioisomerize=True, removeDuplicates=True, getDuplicatedIdx=False, exportCSV=False, outputPath=None, printlogs=True):
    #------------------------
    if getUnrefinedStruct:
        if removeDuplicates:
            print("!!!ERROR: 'removeDuplicates=True' argument goes with 'getUnrefinedStruct=False' only !!!")
            return None
        if getDuplicatedIdx:
            print("!!!ERROR: 'getDuplicatedIdx=True' argument goes with 'getUnrefinedStruct=False' only !!!")  
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
    if isinstance(compounds, str):
        compounds = [compounds]
    #------------------------
    #Validation
    DedupValidated_compounds  = validateComplete(compounds, removeDuplicates=True,  printlogs=False)
    DupValidated_compounds    = validateComplete(compounds, removeDuplicates=False, printlogs=False)
    unvalidated_compounds     = validateComplete(compounds, getInvalidStruct=True,  printlogs=False)
    DuplicatedUnvalidatedIdx  = validateComplete(compounds, removeDuplicates=True,  getDuplicatedIdx=True, printlogs=False)
    if printlogs:
        print("=============================================================================")
        print("VALIDATION")
        if len(DupValidated_compounds) == len(compounds):
            print("-----------------------------------------------------------------------------")
            print("{}/{} structures were successfully validated".format(len(DupValidated_compounds), len(compounds)))   
        else:
            print("-----------------------------------------------------------------------------")
            print("{}/{} structures were successfully validated".format(len(DupValidated_compounds), len(compounds)))   
            print("{}/{} structure(s) was/were unsuccessfully validated and need to be rechecked".format(len(unvalidated_compounds), len(compounds)))
        if removeDuplicates:
            if len(DedupValidated_compounds) == len(DupValidated_compounds):
                print("-----------------------------------------------------------------------------")
                print("No duplicate was found (in {} validated structures)".format(len(DupValidated_compounds)))
                print("=============================================================================")
            else:
                print("-----------------------------------------------------------------------------")
                print("There are {} unique structures filtered from {} initial validated structures".format(len(DedupValidated_compounds), len(DupValidated_compounds)))
                print("=============================================================================")
        else:
            if len(DedupValidated_compounds) == len(DupValidated_compounds):
                print("-----------------------------------------------------------------------------")
                print("No duplicate was found (in {} validated structures)".format(len(DupValidated_compounds)))
                print("=============================================================================")
            else:
                print("-----------------------------------------------------------------------------")
                print ("{}/{} validated structures have at least one duplicates".format(len(DuplicatedUnvalidatedIdx), len(DupValidated_compounds)))
                print("=============================================================================")
    #------------------------
    #Cleaning
    DedupCleaned_compounds = cleanComplete(DedupValidated_compounds, deSalt=deSalt, neutralize=neutralize, removeDuplicates=removeDuplicates, printlogs=False)
    DupCleaned_compounds   = cleanComplete(DupValidated_compounds,   deSalt=deSalt, neutralize=neutralize, removeDuplicates=removeDuplicates, printlogs=False)
    uncleaned_compounds    = cleanComplete(DedupValidated_compounds, deSalt=deSalt, neutralize=neutralize, getUncleanedStruct=True, printlogs=False)
    DuplicatedUncleanedIdx = cleanComplete(DedupValidated_compounds, deSalt=deSalt, neutralize=neutralize, removeDuplicates=True, getDuplicatedIdx=True, printlogs=False)
    if printlogs:
        print("CLEANING")
        if removeDuplicates:
            if len(DedupCleaned_compounds) == len(DedupValidated_compounds):
                print("-----------------------------------------------------------------------------")
                print("{}/{} structures were successfully cleaned".format(len(DedupCleaned_compounds), len(DedupValidated_compounds))) 
            else:
                print("-----------------------------------------------------------------------------")
                print("{}/{} structures were successfully cleaned".format(len(DedupCleaned_compounds), len(DedupValidated_compounds)))   
                print("{}/{} structure(s) was/were unsuccessfully cleaned and need to be rechecked".format(len(DedupValidated_compounds)-len(DedupCleaned_compounds), len(DedupValidated_compounds)))  
            if len(DedupCleaned_compounds) == len(DupCleaned_compounds):
                print("-----------------------------------------------------------------------------")
                print("No duplicate was found (in {} cleaned structures)".format(len(DupCleaned_compounds)))
                print("=============================================================================")
            else:
                print("-----------------------------------------------------------------------------")
                print("There are {} unique structures filtered from {} initial cleaned structures".format(len(DedupCleaned_compounds), len(DupCleaned_compounds)))
                print("=============================================================================")
        else:
            if len(DupCleaned_compounds) == len(DupValidated_compounds):
                print("-----------------------------------------------------------------------------")
                print("{}/{} structures were successfully cleaned".format(len(DupCleaned_compounds), len(DupValidated_compounds))) 
            else:
                print("-----------------------------------------------------------------------------")
                print("{}/{} structures were successfully cleaned".format(len(DupCleaned_compounds), len(DupValidated_compounds)))   
                print("{}/{} structure(s) was/were unsuccessfully cleaned and need to be rechecked".format(len(DupValidated_compounds)-len(DupCleaned_compounds), len(DupValidated_compounds)))  
            if len(DedupCleaned_compounds) == len(DupCleaned_compounds):
                print("-----------------------------------------------------------------------------")
                print("No duplicate was found (in {} cleaned structures)".format(len(DupCleaned_compounds)))
                print("=============================================================================")
            else:
                print("-----------------------------------------------------------------------------")
                if DuplicatedUncleanedIdx['idx'][0] != 'na':
                    print ("{}/{} cleaned structures have at least one duplicates".format(len(DuplicatedUncleanedIdx), len(DupCleaned_compounds)))
                else:
                    print("No duplicate was found (in {} cleaned structures)".format(len(DupCleaned_compounds)))
                print("=============================================================================")
    #------------------------
    #Normalization
    DedupNormalized_compounds = normalizeComplete(DedupCleaned_compounds, deTautomerize=deTautomerize, deSterioisomerize=deSterioisomerize, removeDuplicates=removeDuplicates, printlogs=False)
    DupNormalized_compounds   = normalizeComplete(DupCleaned_compounds,   deTautomerize=deTautomerize, deSterioisomerize=deSterioisomerize, removeDuplicates=removeDuplicates, printlogs=False)
    unnormalized_compounds    = normalizeComplete(DedupCleaned_compounds, deTautomerize=deTautomerize, deSterioisomerize=deSterioisomerize, getUnnormalizedStruct=True, printlogs=False)
    DuplicatedUnNormalizedIdx = normalizeComplete(DedupCleaned_compounds, deTautomerize=deTautomerize, deSterioisomerize=deSterioisomerize, removeDuplicates=True, getDuplicatedIdx=True, printlogs=False)
    if printlogs:
        print("NORMALIZATION")
        if removeDuplicates:
            if len(DedupNormalized_compounds) == len(DedupCleaned_compounds):
                print("-----------------------------------------------------------------------------")
                print("{}/{} structures were successfully normalized".format(len(DedupNormalized_compounds), len(DedupCleaned_compounds))) 
            else:
                print("-----------------------------------------------------------------------------")
                print("{}/{} structures were successfully normalized".format(len(DedupNormalized_compounds), len(DedupCleaned_compounds)))   
                print("{}/{} structure(s) were unsuccessfully normalized and need to be rechecked".format(len(DedupCleaned_compounds)-len(DedupNormalized_compounds), len(DedupCleaned_compounds)))  
            if len(DedupNormalized_compounds) == len(DupNormalized_compounds):
                print("-----------------------------------------------------------------------------")
                print("No duplicate was found (in {} normalized structures)".format(len(DupNormalized_compounds)))
                print("=============================================================================")
            else:
                print("-----------------------------------------------------------------------------")
                print("There are {} unique structures filtered from {} initial normalized structures".format(len(DedupNormalized_compounds), len(DupNormalized_compounds)))
                print("=============================================================================")
        else:
            if len(DupNormalized_compounds) == len(DupCleaned_compounds):
                print("-----------------------------------------------------------------------------")
                print("{}/{} structures were successfully normalized".format(len(DupNormalized_compounds), len(DupCleaned_compounds))) 
            else:
                print("-----------------------------------------------------------------------------")
                print("{}/{} structures were successfully normalized".format(len(DupNormalized_compounds), len(DupCleaned_compounds)))   
                print("{}/{} structure(s) was/were unsuccessfully normalized and need to be rechecked".format(len(DupCleaned_compounds)-len(DupNormalized_compounds), len(DupCleaned_compounds)))  
            if len(DedupNormalized_compounds) == len(DupNormalized_compounds):
                print("-----------------------------------------------------------------------------")
                print("No duplicate was found (in {} normalized structures)".format(len(DupNormalized_compounds)))
                print("=============================================================================")
            else:
                print("-----------------------------------------------------------------------------")
                if DuplicatedUnNormalizedIdx['idx'][0] != 'na':
                    print ("{}/{} normalized structures have at least one duplicates".format(len(DuplicatedUnNormalizedIdx), len(DupNormalized_compounds)))
                else:
                    print("No duplicate was found (in {} normalized structures)".format(len(DupNormalized_compounds)))
                print("=============================================================================")
    #------------------------
    DedupRefined_compounds    = DedupNormalized_compounds
    DupRefined_compounds      = DupNormalized_compounds
    unrefined_compounds       = pd.concat([unvalidated_compounds, uncleaned_compounds,unnormalized_compounds], ignore_index=True)
    DuplicatedUnrefinedIdx_df = pd.concat([DuplicatedUnvalidatedIdx, DuplicatedUncleanedIdx, DuplicatedUnNormalizedIdx], ignore_index=True)
    if printlogs:
        print("REFINEMENT SUMMARY")
        if removeDuplicates:
            compounds_ = molDeduplicate(compounds, printlogs=False)
            if len(DedupNormalized_compounds) == len(compounds_):
                print("-----------------------------------------------------------------------------")
                print("{}/{} deduplicated structures were successfully refined".format(len(DedupNormalized_compounds), len(compounds_))) 
            else:
                print("-----------------------------------------------------------------------------")
                print("{}/{} deduplicated structures were successfully refined".format(len(DedupNormalized_compounds), len(compounds_)))   
                print("{}/{} deduplicated structure(s) was/were unsuccessfully refined and need to be rechecked".format(len(compounds_)-len(DedupNormalized_compounds), len(compounds_)))  
            if len(DedupRefined_compounds) == len(DupRefined_compounds):
                print("-----------------------------------------------------------------------------")
                print("No duplicate was found (in {} refined structures)".format(len(DedupRefined_compounds)))
                print("=============================================================================")
            else:
                print("-----------------------------------------------------------------------------")
                print("There are {} unique structures filtered from {} initial refined structures".format(len(DedupRefined_compounds), len(DupRefined_compounds)))
                print("=============================================================================")
        else:
            if len(DupNormalized_compounds) == len(compounds):
                print("-----------------------------------------------------------------------------")
                print("{}/{} structures were successfully refined (duplicates existing)".format(len(DupNormalized_compounds), len(compounds))) 
            else:
                print("-----------------------------------------------------------------------------")
                print("{}/{} structures were successfully refined (duplicates existing)".format(len(DupNormalized_compounds), len(compounds)))   
                print("{}/{} structure(s) was/were unsuccessfully refined and need to be rechecked (duplicates existing)".format(len(compounds)-len(DupNormalized_compounds), len(compounds)))  
            if len(DedupNormalized_compounds) == len(DupNormalized_compounds):
                print("-----------------------------------------------------------------------------")
                print("No duplicate was found (in {} refined structures)".format(len(DedupRefined_compounds)))
                print("=============================================================================")
            else:
                print("-----------------------------------------------------------------------------")
                print ("{}/{} refined structures have at least one duplicates".format(len(DuplicatedUnrefinedIdx_df), len(DupRefined_compounds)))
                print("=============================================================================")
    #------------------------
    if getUnrefinedStruct:
        df1 = unrefined_compounds
        if exportCSV:
            filePath = outputPath + "UnrefinedList.csv"
            if os.path.isdir(outputPath):
                df1.to_csv(filePath, index=False)
            else:
                os.makedirs(outputPath)
                df1.to_csv(filePath, index=False)
        else:
            return df1
    else:
        if removeDuplicates:
            if getDuplicatedIdx:
                df2 = DuplicatedUnrefinedIdx_df
                if exportCSV:
                    filePath = outputPath + "DuplicatedRefinedIdxList.csv"
                    if os.path.isdir(outputPath):
                        df2.to_csv(filePath, index=False)
                    else:
                        os.makedirs(outputPath)
                        df2.to_csv(filePath, index=False)
                else:
                    return df2
            else:
                df2 = DedupRefined_compounds
                if exportCSV:
                    filePath = outputPath + "DeduplicatedRefinedList.csv"
                    if os.path.isdir(outputPath):
                        df2.to_csv(filePath, index=False)
                    else:
                        os.makedirs(outputPath)
                        df2.to_csv(filePath, index=False)
                else:
                    return df2
        else:
            df3 = DupRefined_compounds
            if exportCSV:
                filePath = outputPath + "DuplicatedRefinedList.csv"
                if os.path.isdir(outputPath):
                    df3.to_csv(filePath, index=False)
                else:
                    os.makedirs(outputPath)
                    df3.to_csv(filePath, index=False)
            else:
                return df3            

                
      