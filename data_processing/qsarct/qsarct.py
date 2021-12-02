from rdkit import Chem
import pandas as pd

#==========================================================
def validate(compounds, get_failed_structure=False):
    success_list, fail_list = [], []
    for c_index in range(len(compounds)):
        mol = Chem.MolFromSmiles(compounds[c_index])
        if mol != None:
            success_list.append(Chem.MolToSmiles(mol))
        else:
            fail_list.append(compounds[c_index])
    if len(fail_list) > 0:
        print("Succeeded to validate {}/{} structures".format(len(success_list), len(compounds)))
        print("Failed to validate {} structures".format(len(fail_list)))
        print("Set 'get_failed_structure=True' to get the list of failed structures")
    else:
        print("Succeeded to validate {} structures".format(len(success_list)))
    if get_failed_structure:
        return fail_list
    else:
        return success_list

#==========================================================
def remove_dup(compounds, get_dup_list=False):
    dup_list =[]
    new_compounds = sorted(set(compounds), key=compounds.index)
    if len(new_compounds) == len(compounds):
        print("No duplicate is found")
        unique_list = new_compounds
    else:
        unique_list = []
        for compound in compounds:
            occuring = [i for i, c in enumerate(compounds) if c == compound]
            if len(occuring) > 1:
                dup_list.append(compound)
                if compound not in unique_list:  
                    unique_list.append(compound)
            else:
                unique_list.append(compound)
        dup_list = sorted(set(dup_list), key=dup_list.index)     
        print ("{}/{} structures have at least one duplicates".format(len(dup_list),len(compounds)))
        print ("After removing duplicates, there are {} structures".format(len(unique_list)))
        print ("Set 'get_dup_list=False' to get the list of duplicates")
    if get_dup_list:
        return dup_list
    else:    
        return unique_list
    
#==========================================================
def remove_iom(compounds, path=None, get_i=False, get_s=False, get_m=False):
    excluded_list = ['Cr', 'Co', 'Cl', 'Cd', 'Cn', 'Cs', 'Ce', 'Cf', 'Cm']
    success_list_i, success_list_s, success_list_m = [], [], []
    fail_list_i, fail_list_s, fail_list_m = [], [], []
    # Remove non-C structures
    for c_index in range(len(compounds)):
        if compounds[c_index].find('C') == -1:
            fail_list_i.append(compounds[c_index])
        else:
            result = map(lambda c: compounds[c_index].find(c), excluded_list)
            if sum(list(result)) > len(list(result))/(-1):
                fail_list_i.append(compounds[c_index])
            else:
                success_list_i.append(compounds[c_index])
    # Remove salt 
    for c_index in range(len(compounds)):
        if compounds[c_index].find('.') == -1:
            success_list_s.append(compounds[c_index])
        else:
            fail_list_s.append(compounds[c_index])
    # Remove metal-containing compounds
    metals_list = open("./qsarct/metals.txt").read().split(',')
    for c_index in range(len(compounds)):
        result = map(lambda c: compounds[c_index].find(c), metals_list)
        if sum(list(result)) > len(list(result))/(-1):
            fail_list_m.append(compounds[c_index])
        else:
            success_list_m.append(compounds[c_index])
    # Get substep results
    if get_i and path is not None:
        list_i = success_list_i + fail_list_i
        label_i = ['succeeded']*len(success_list_i) + ['failed']*len(fail_list_i)
        pd.DataFrame(zip(list_i, label_i), columns=['SMILES', 'Status']).to_csv(path, index=False)
    if get_s and path is not None:
        list_o = success_list_s + fail_list_s
        label_o = ['succeeded']*len(success_list_s) + ['failed']*len(fail_list_s)
        pd.DataFrame(zip(list_o, label_o), columns=['SMILES', 'Status']).to_csv(path, index=False)
    if get_m and path is not None:
        list_m = success_list_m + fail_list_m
        label_m = ['succeeded']*len(success_list_m) + ['failed']*len(fail_list_m)
        pd.DataFrame(zip(list_m, label_m), columns=['SMILES', 'Status']).to_csv(path, index=False)
    # Get final result
    refined_list = list(set.intersection(set(success_list_i), set(success_list_s), set(success_list_m)))
    # Print results
    print("{}/{} inorganic structures were removed".format(len(compounds) - len(success_list_i), len(compounds)))
    print("{}/{} salt structures were removed".format(len(compounds) - len(success_list_s), len(compounds)))
    print("{}/{} metal-containing structures were removed".format(len(compounds) - len(success_list_m), len(compounds)))
    print("===============================================")
    print("{}/{} structures were successfully refined".format(len(refined_list), len(compounds)))
    return refined_list

#==========================================================
def full_refine(compounds, get_failed_compounds=False):
    validated     = validate(compounds)
    print("===============================================")
    invalidated   = validate(compounds, get_failed_structure=True)
    print("===============================================")
    dup_filtered  = remove_dup(validated)
    print("===============================================")
    iom_filtered  = remove_iom(dup_filtered)
    if get_failed_compounds:
        failed_compounds = list(set(validated + invalidated)  - set(iom_filtered))
        return failed_compounds
    else:
        return dup_filtered