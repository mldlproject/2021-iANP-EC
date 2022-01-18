# Import libraries
# Import libraries
import pandas as pd
from chemtonic.curation.refinement import refineComplete

#============================================================
# Load active/inactive compounds
npass_compounds = pd.read_csv("https://raw.githubusercontent.com/mldlproject/2019-PlantDB_review/master/data/NPASS_SMILE.csv", header=None)
npass_compounds = npass_compounds[0].tolist()

#============================================================
# Fullly refine
refined_npass_compounds = refineComplete(npass_compounds)
pd.DataFrame(refined_npass_compounds, columns = ['SMILES']).to_csv("../data/refined_data/refined_npass.csv", index=False)
