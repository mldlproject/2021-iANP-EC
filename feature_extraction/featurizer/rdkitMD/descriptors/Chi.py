import pandas as pd
from rdkit.Chem import GraphDescriptors

colnames = ['BalabanJ', 'BertzCT',
            'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v']
def get_Chi(mol):
    BalabanJ = GraphDescriptors.BalabanJ(mol)
    BertzCT = GraphDescriptors.BertzCT(mol)
    BalBertz_df = [BalabanJ, BertzCT]

    Chi0  = GraphDescriptors.Chi0(mol)
    Chi0n = GraphDescriptors.Chi0n(mol)
    Chi0v = GraphDescriptors.Chi0v(mol)
    Chi1  = GraphDescriptors.Chi1(mol)
    Chi1n = GraphDescriptors.Chi1n(mol)
    Chi1v = GraphDescriptors.Chi1v(mol)
    Chi2n = GraphDescriptors.Chi2n(mol)
    Chi2v = GraphDescriptors.Chi2v(mol)
    Chi3n = GraphDescriptors.Chi3n(mol)
    Chi3v = GraphDescriptors.Chi3v(mol)
    Chi4n = GraphDescriptors.Chi4n(mol)
    Chi4v = GraphDescriptors.Chi4v(mol)
    Chi = [Chi0, Chi0n, Chi0v, Chi1, Chi1n, Chi1v, Chi2n, Chi2v, Chi3n, Chi3v, Chi4n, Chi4v]
    
    ChiBalBertz = BalBertz_df + Chi
    ChiBalBertz_df = pd.DataFrame(ChiBalBertz).T
    ChiBalBertz_df.columns = colnames
    return ChiBalBertz_df