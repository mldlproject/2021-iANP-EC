from rdkit.Chem import Fragments
import pandas as pd

colnames = ['fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 
            'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 
            'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 
            'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 
            'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 
            'fr_isothiocyan', 'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 
            'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 
            'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 
            'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea']

def get_fragments(mol):
    fr_Al_COO = Fragments.fr_Al_COO(mol)
    fr_Al_OH  = Fragments.fr_Al_OH(mol)
    fr_Al_OH_noTert = Fragments.fr_Al_OH_noTert(mol)
    fr_ArN = Fragments.fr_ArN(mol)
    fr_Ar_COO = Fragments.fr_Ar_COO(mol)
    fr_Ar_N = Fragments.fr_Ar_N(mol)
    fr_Ar_NH = Fragments.fr_Ar_NH(mol)
    fr_Ar_OH = Fragments.fr_Ar_OH(mol)
    fr_COO = Fragments.fr_COO(mol)
    fr_COO2 = Fragments.fr_COO2(mol)
    fr1 = [fr_Al_COO, fr_Al_OH, fr_Al_OH_noTert, fr_ArN, fr_Ar_COO, fr_Ar_N, fr_Ar_NH, fr_Ar_OH, fr_COO, fr_COO2]
    #10

    fr_C_O = Fragments.fr_C_O(mol)
    fr_C_O_noCOO = Fragments.fr_C_O_noCOO(mol)
    fr_C_S = Fragments.fr_C_S(mol)
    fr_HOCCN = Fragments.fr_HOCCN(mol)
    fr_Imine = Fragments.fr_Imine(mol)
    fr_NH0 = Fragments.fr_NH0(mol)
    fr_NH1 = Fragments.fr_NH1(mol)
    fr_NH2 = Fragments.fr_NH2(mol)
    fr_N_O = Fragments.fr_N_O(mol)
    fr_Ndealkylation1 = Fragments.fr_Ndealkylation1(mol)
    fr2 = [fr_C_O, fr_C_O_noCOO, fr_C_S, fr_HOCCN, fr_Imine, fr_NH0, fr_NH1, fr_NH2, fr_N_O, fr_Ndealkylation1]
    #10

    fr_Ndealkylation2 = Fragments.fr_Ndealkylation2(mol)
    fr_Nhpyrrole = Fragments.fr_Nhpyrrole(mol)
    fr_SH = Fragments.fr_SH(mol)
    fr_aldehyde = Fragments.fr_aldehyde(mol)
    fr_alkyl_carbamate = Fragments.fr_alkyl_carbamate(mol)
    fr_alkyl_halide = Fragments.fr_alkyl_halide(mol)
    fr_allylic_oxid = Fragments.fr_allylic_oxid(mol)
    fr_amide = Fragments.fr_amide(mol)
    fr_amidine = Fragments.fr_amidine(mol)
    fr_aniline = Fragments.fr_aniline(mol)
    fr3 = [fr_Ndealkylation2, fr_Nhpyrrole, fr_SH, fr_aldehyde, fr_alkyl_carbamate, fr_alkyl_halide, fr_allylic_oxid, fr_amide, fr_amidine, fr_aniline]
    #10

    fr_aryl_methyl = Fragments.fr_aryl_methyl(mol)
    fr_azide = Fragments.fr_azide(mol)
    fr_azo = Fragments.fr_azo(mol)
    fr_barbitur = Fragments.fr_barbitur(mol)
    fr_benzene = Fragments.fr_benzene(mol)
    fr_benzodiazepine = Fragments.fr_benzodiazepine(mol)
    fr_bicyclic = Fragments.fr_bicyclic(mol)
    fr_diazo = Fragments.fr_diazo(mol)
    fr_dihydropyridine = Fragments.fr_dihydropyridine(mol)
    fr_epoxide = Fragments.fr_epoxide(mol)
    fr4 = [fr_aryl_methyl, fr_azide, fr_azo, fr_barbitur, fr_benzene, fr_benzodiazepine, fr_bicyclic, fr_diazo, fr_dihydropyridine, fr_epoxide]
    #10

    fr_ester = Fragments.fr_ester(mol)
    fr_ether = Fragments.fr_ether(mol)
    fr_furan = Fragments.fr_furan(mol)
    fr_guanido = Fragments.fr_guanido(mol)
    fr_halogen = Fragments.fr_halogen(mol)
    fr_hdrzine = Fragments.fr_hdrzine(mol)
    fr_hdrzone = Fragments.fr_hdrzone(mol)
    fr_imidazole = Fragments.fr_imidazole(mol)
    fr_imide = Fragments.fr_imide(mol)
    fr_isocyan = Fragments.fr_isocyan(mol)
    fr5 = [fr_ester, fr_ether, fr_furan, fr_guanido, fr_halogen, fr_hdrzine, fr_hdrzone, fr_imidazole, fr_imide, fr_isocyan]
    #10

    fr_isothiocyan = Fragments.fr_isothiocyan(mol)
    fr_ketone = Fragments.fr_ketone(mol)
    fr_ketone_Topliss = Fragments.fr_ketone_Topliss(mol)
    fr_lactam = Fragments.fr_lactam(mol)
    fr_lactone = Fragments.fr_lactone(mol)
    fr_methoxy = Fragments.fr_methoxy(mol)
    fr_morpholine = Fragments.fr_morpholine(mol)
    fr_nitrile = Fragments.fr_nitrile(mol)
    fr_nitro = Fragments.fr_nitro(mol)
    fr_nitro_arom = Fragments.fr_nitro_arom(mol)
    fr6 = [fr_isothiocyan, fr_ketone, fr_ketone_Topliss, fr_lactam, fr_lactone, fr_methoxy, fr_morpholine, fr_nitrile, fr_nitro, fr_nitro_arom]
    #10

    fr_nitro_arom_nonortho = Fragments.fr_nitro_arom_nonortho(mol)
    fr_nitroso = Fragments.fr_nitroso(mol)
    fr_oxazole = Fragments.fr_oxazole(mol)
    fr_oxime = Fragments.fr_oxime(mol)
    fr_para_hydroxylation = Fragments.fr_para_hydroxylation(mol)
    fr_phenol = Fragments.fr_phenol(mol)
    fr_phenol_noOrthoHbond = Fragments.fr_phenol_noOrthoHbond(mol)
    fr_phos_acid = Fragments.fr_phos_acid(mol)
    fr_phos_ester = Fragments.fr_phos_ester(mol)
    fr_piperdine = Fragments.fr_piperdine(mol)
    fr7 = [fr_nitro_arom_nonortho, fr_nitroso, fr_oxazole, fr_oxime, fr_para_hydroxylation, fr_phenol, fr_phenol_noOrthoHbond, fr_phos_acid, fr_phos_ester, fr_piperdine]
    #10

    fr_piperzine = Fragments.fr_piperzine(mol)
    fr_priamide = Fragments.fr_priamide(mol)
    fr_prisulfonamd = Fragments.fr_prisulfonamd(mol)
    fr_pyridine = Fragments.fr_pyridine(mol)
    fr_quatN = Fragments.fr_quatN(mol)
    fr_sulfide = Fragments.fr_sulfide(mol)
    fr_sulfonamd = Fragments.fr_sulfonamd(mol)
    fr_sulfone = Fragments.fr_sulfone(mol)
    fr_term_acetylene = Fragments.fr_term_acetylene(mol)
    fr_tetrazole = Fragments.fr_tetrazole(mol)
    fr_thiazole = Fragments.fr_thiazole(mol)
    fr_thiocyan = Fragments.fr_thiocyan(mol)
    fr_thiophene = Fragments.fr_thiophene(mol)
    fr_unbrch_alkane = Fragments.fr_unbrch_alkane(mol)
    fr_urea = Fragments.fr_urea(mol)
    fr8 = [fr_piperzine, fr_priamide, fr_prisulfonamd, fr_pyridine, fr_quatN, fr_sulfide, fr_sulfonamd, fr_sulfone, fr_term_acetylene, fr_tetrazole, fr_thiazole, fr_thiocyan, fr_thiophene, fr_unbrch_alkane, fr_urea]
    #15

    fr = fr1 + fr2 + fr3 + fr4 + fr5 + fr6 + fr7 + fr8
    fr_df = pd.DataFrame(fr).T
    fr_df.columns = colnames

    return fr_df


