import drugdiscovery.preprocessing as pp
import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    source = pd.read_csv('data/data_dups_removed.csv',index_col = 0)
    data = source.copy()

    print('Generating Morgan fingerprint data')
    smiles = data['SMILES'].values
    mf = pp.MorganFingerprints(smiles)
    X = mf.transform()
    df = pd.DataFrame(data[['DSSTox_CID','SMILES']])
    N,p = X.shape
    for i in tqdm(range(0,p)):
        df['fp_'+str(i+1)] = X[:,i]

    df.to_csv('data/morganfingerprint.csv')

    print('Generating Murcko Scaffold data')
    smiles = data['SMILES'].values
    ms = pp.MurckoScaffold(smiles)
    X = ms.transform()
    df = pd.DataFrame(data[['DSSTox_CID','SMILES']])
    df['MuckoScaffold'] = X

    df.to_csv('data/murckoscaffoldfingerprint.csv')
