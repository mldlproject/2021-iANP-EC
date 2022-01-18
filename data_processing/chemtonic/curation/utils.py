from rdkit import Chem
import pandas as pd 

#==========================================================
# validate SMILES
def molStructVerify(compounds, getFailedStruct=False, getFailedStructIdx=False, printlogs=True):
    #------------------------
    if getFailedStructIdx:
        if getFailedStruct == False:
            print("!!!ERROR: 'getFailedStructIdx=True' argument goes with 'getFailedStruct=True'!!!")
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
    VerifiedList, UnverifiedList, UnverifiedIdxList = [], [], []
    for c_index in range(len(compounds)):
        mol = Chem.MolFromSmiles(compounds[c_index])
        if mol != None:
            VerifiedList.append(Chem.MolToSmiles(mol))
        else:
            UnverifiedList.append(compounds[c_index])
            UnverifiedIdxList.append(c_index)
    if printlogs:
        if len(UnverifiedList) > 0:
            print("Succeeded to verify {}/{} structures".format(len(VerifiedList), len(compounds)))
            print("Failed to verify {} structures".format(len(UnverifiedList)))
            print("Set 'getFailedStruct=True' to get the list of failed structures")
        else:
            print("Succeeded to verify {}/{} structures".format(len(VerifiedList), len(compounds)))
    #------------------------
    if getFailedStruct:
        if getFailedStructIdx:
            return UnverifiedList, UnverifiedIdxList
        else:
            return UnverifiedList
    else:
        return VerifiedList
    
#==========================================================
# remove dupplicates
def molDeduplicate(compounds, getDuplicates=False, getDuplicatesIdx=False, printlogs=True):
    #------------------------
    if getDuplicatesIdx:
        if getDuplicates == False:
            print("!!!ERROR: 'getDuplicatesIdx=True' argument goes with 'getDuplicates=True'!!!")
            return None
    #------------------------
    if isinstance(compounds, pd.core.series.Series):
        compounds_ = compounds.tolist()
    if isinstance(compounds, pd.core.frame.DataFrame):
        compounds_ = compounds.iloc[:,0].tolist()
    if isinstance(compounds, str):
        compounds_ = [compounds]
    if isinstance(compounds, list):
        compounds_ = compounds
    #------------------------
    compounds__ = molStructVerify(compounds_, printlogs=False)
    Unverified_count = len(molStructVerify(compounds_, getFailedStruct=True, printlogs=False))
    SortedFilteredList = sorted(set(compounds__), key=compounds__.index)
    #------------------------
    if len(SortedFilteredList) == len(compounds__):
        UniqueList = SortedFilteredList
        DuplicatedIdxList = [('na', 'na')] 
        DuplicatedList = ['na']
    else:
        UniqueList, DuplicatedList, DuplicatedIdxList = [], [], []
        idx = 0
        for compound in compounds__:
            occuring = [i for i, c in enumerate(compounds__) if c == compound]
            if len(occuring) > 1:
                DuplicatedIdxList.append((idx, occuring))
                DuplicatedList.append(compound)
                if compound not in UniqueList:  
                    UniqueList.append(compound)
            else:
                UniqueList.append(compound)
            idx += 1
    if printlogs:
        if Unverified_count > 0:
            print("Failed to verify {} structures".format(Unverified_count))
            print("Use function 'molValidate' and set 'getFailedStruct=True' to get the list of unverified structures")
            if len(SortedFilteredList) == len(compounds__):
                print("No duplicate is found (in {} verified structures)".format(len(compounds__)))
            else:
                print ("{}/{} structures have at least one duplicates".format(len(DuplicatedList), len(compounds__)))
                print ("There are {} unique structures filtered from {} initial structures".format(len(UniqueList), len(compounds__)))
                print ("Set 'getDuplicates=True' to get the list of duplicated structures")
                print ("Set 'getDuplicates=True' and 'getDuplicatesIndex'=True to get the list of duplicated structures with detailed indices")
        else:
            print("Succeeded to verify {}/{} structures".format(len(compounds__), len(compounds)))
            if len(SortedFilteredList) == len(compounds__):
                print("No duplicate is found (in {} verified structures)".format(len(compounds__)))
            else:
                print ("{}/{} structures have at least one duplicates".format(len(DuplicatedList), len(compounds__)))
                print ("There are {} unique structures filtered from {} initial structures".format(len(UniqueList), len(compounds__)))
                print ("Set 'getDuplicates=True' to get the list of duplicated structures")
                print ("Set 'getDuplicates=True' and 'getDuplicatesIndex'=True to get the list of duplicated structures with detailed indices")
    #------------------------
    if getDuplicates:
        if getDuplicatesIdx:
            return DuplicatedList, DuplicatedIdxList
        else:
                return DuplicatedList    
    else: 
        return UniqueList

#==========================================================
# neutralize_atoms SMILES of charged compounds (Author: Noel O’Boyle (Vincent Scalfani adapted code for RDKit))
def neutralize_atoms(mol):
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    #------------------------
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            if hcount != 0:
                atom.SetFormalCharge(0)
                atom.SetNumExplicitHs(hcount - chg)
                atom.UpdatePropertyCache()
            else:
                atom.UpdatePropertyCache()
    #------------------------
    return mol

#==========================================================
# Get intersection
def intersectionList(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    #------------------------
    return lst3