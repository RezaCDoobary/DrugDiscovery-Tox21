################
# Author: Reza C Doobary
#
# preprocess.py
#
# The script performs the preprocessing steps after performing the initial clean to the data for the Tox-21 challenge.
#
#   In this script the following is generated:
#            1. The hydrogen atoms a put back into the smiles (used for BoW)
#            2. Morgan fingerprints are generated for the smiles.
#            3. Murcko Scaffolds are generated for the smiles.
#            4. Morgan fingerprints are generated for the scaffolds.
#            5. Molecular descriptors are computed for the smiles.
#            6. Dice similarties are generated for the smiles with certain molecules known to have toxic properties.
#    
#    Note - that not all of these datasets are used in the modelling portion of the exercise.
#
################

# Imports
import drugdiscovery.preprocessing as pp
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

def add_hydrogen_to_source_data(data:pd.core.frame.DataFrame)->str:
    """
    Takes the dataframe containing the smiles, adds the hydrogen back in, saves it in the data folder 
    with a new name. The new filename is the output to the function.
    """
    def addHs(smile):
        m=Chem.MolFromSmiles(smile)
        m=Chem.AddHs(m)
        return Chem.MolToSmiles(m)

    data['SMILES'] = data['SMILES'].map(lambda x: addHs(x))
    output_file = 'data/data_dups_removed_with_H.csv'
    data.to_csv(output_file)
    return output_file


def generate_morgan_fingerprints_from_source(data:pd.core.frame.DataFrame, radius:int)->str:
    """
    Generates the morgan fingerprints.
    """
    print('Generate morgan fignerprint from source : ',radius)

    smiles = data['SMILES'].values
    mf = pp.MorganFingerprints(smiles)
    X = mf.transform(radius)
    df = pd.DataFrame(data[['DSSTox_CID','SMILES']])
    N,p = X.shape
    for i in tqdm(range(0,p)):
        df['fp_'+str(i+1)] = X[:,i]

    output_file = 'data/morgan_fingerprint_from_source_'+str(radius)+'.csv'
    df.to_csv(output_file)
    return output_file

def generate_murckoscaffold_from_source(data:pd.core.frame.DataFrame)->str:
    """
    Generates the murcko scaffold.
    """
    print('Generating Murcko Scaffold data')
    smiles = data['SMILES'].values
    ms = pp.MurckoScaffold(smiles)
    X = ms.transform()
    df = pd.DataFrame(data[['DSSTox_CID','SMILES']])
    df['MuckoScaffold'] = X

    output_file = 'data/murckoscaffold.csv'
    df.to_csv(output_file)
    return output_file

def generate_morgan_fingerprints_from_scaffold(data:pd.core.frame.DataFrame, radius:int)->str:
    """
    Generates the fingerprints for the scaffolds.
    """
    print('Generating fingerprints for Murcko Scaffold data : ',radius)

    
    scaffold_smiles = data['MuckoScaffold'].values
    mf = pp.MorganFingerprints(scaffold_smiles)
    X = mf.transform(radius)
    
    df = pd.DataFrame(data[['DSSTox_CID','SMILES','MuckoScaffold']])
    N,p = X.shape
    for i in tqdm(range(0,p)):
        df['fp_'+str(i+1)] = X[:,i]

    output_file = 'data/morgan_fingerprint_from_scaffold_'+str(radius)+'.csv'
    df.to_csv(output_file)
    return output_file

def generate_molecular_descriptors_from_source(source:pd.core.frame.DataFrame)->str:
    """
    Generates the molecular descriptors given the smiles.
    """
    print('Generating molecular descriptors for source data')
    df = pd.DataFrame(source['SMILES'])
    N = len(df)
    for des in pp.all_descriptors:
        df[des] = np.nan

    for idx in tqdm(df.index):
        smile = df.loc[idx]['SMILES']
        mol = Chem.MolFromSmiles(smile) 
        moldes = pp.MolecularDescriptors(mol)
        res = moldes.compute_all(mol)
        
        for key, value in res.items():
            df.at[idx,key] = value

    output_file = 'data/molecular_descriptors_from_source.csv'
    df.to_csv(output_file)
    return output_file

def compute_similarity_scaffold_known_toxic()->None:
    """
    Generated the Dice similarties between the smiles and known toxic molecules.
    """
    knownToxic = ['C1=CC=C(C=C1)N','C1=CC=C(C=C1)CBr','C1=CC=C(C=C1)CI','C1=CC=C(C=C1)CCl','C1=CC=C(C=C1)CF',\
             'CNN','C1=CC=C(C=C1)[N+](=O)[O-]','NN=O','C1=CC=C2C(=C1)C=CC3=CC=CC=C32','C1=CC(=O)C=CC1=O',\
             'C(CCl)SCCCl','C(=S)(N)N']
    mf = pp.MorganFingerprints(knownToxic)
    knownTox = np.array(mf.transform(radius = 10),int)

    knownTox_fa = mf.fingerprint_array

    df = pd.read_csv('data/morgan_fingerprint_from_scaffold_10.csv',index_col = 0)
    scaffolds = df['MuckoScaffold'].values
    mf = pp.MorganFingerprints(scaffolds)
    scaffold_fa = np.array(mf.transform(radius = 10),int)

    scaffold_fa = mf.fingerprint_array

    from rdkit import DataStructs
    result = [[DataStructs.DiceSimilarity(scaf,kTfa) for kTfa in knownTox_fa] for scaf in scaffold_fa]

    df = pd.DataFrame(np.array(result), columns=knownToxic,index=df.index)
    df.to_csv('data/known_toxic.csv')

def clean_data_zeros_nan(filename):
    """
    A final function that goes though all the generated files and removves columns which are either complete zero or have a single null in them.
    """
    df = pd.read_csv(filename,index_col = 0)
    res = df.apply(lambda x: (x == 0).all()).to_dict()

    for k,v in res.items():
        if v:
            del df[k]

    res = df.apply(lambda x: (x.isnull()).any())
    for k,v in res.items():
        if v:
            del df[k]

    df.to_csv(filename)




if __name__ == "__main__":
    files = []

    # step 0: get the data.
    source = pd.read_csv('data/data_dups_removed.csv',index_col = 0)
    data = source.copy()

    # step 1. Ad hydrogen back.
    add_hydrogen_to_source_data(data)

    # step 2. Morgan fingerprints.
    files.append(generate_morgan_fingerprints_from_source(data, 10))
    
    # step 3. Murcko Scaffolds
    files.append(generate_murckoscaffold_from_source(data))

    # step 4. Fingerprints for the scaffold
    scaffold = pd.read_csv('data/murckoscaffold.csv',index_col = 0)
    scaffold['MuckoScaffold'] = scaffold['MuckoScaffold'].fillna(scaffold['SMILES'])
    files.append(generate_morgan_fingerprints_from_scaffold(scaffold, 10))
    
    # step 5. Molecular descriptors.
    source = pd.read_csv('data/data_dups_removed.csv',index_col = 0)
    files.append(generate_molecular_descriptors_from_source(source))

    # step 6. Get toxicophores similarities.
    compute_similarity_scaffold_known_toxic()

    # step 7. Clean generated files.
    for file in files:
        clean_data_zeros_nan(file)



