import numpy as np
import pandas as pd 
from collections import Counter
from rdkit.Chem.Scaffolds import MurckoScaffold as MS
from rdkit import Chem
from rdkit.Chem import AllChem

class CreateSymbolDataset(object):
    def __init__(self, smiles):
        self.smiles = smiles
        self.max_size = self._get_max_size()
        self.N = len(self.smiles)
        
    def _get_max_size(self):
        max_size = max(map(len,self.smiles))
        return max_size
    
    def _get_list_symbols(self):
        symbols = set()
        for smile in self.smiles:
            symbols = symbols.union(set(list(smile)))
        symbols = list(symbols)

        sym_idx = {symbols[i]:i for i in range(0,len(symbols))}
        self.symbols = symbols
        self.sym_idx = sym_idx

    
    def get_symbols(self):
        return self.symbols
    
    def get_symbol_index(self):
        return self.sym_idx


class BagOfWords(CreateSymbolDataset):
    def __init__(self,smiles):
        super(BagOfWords, self).__init__(smiles)
    
    def fit(self):
        self._get_list_symbols()
        self.symbols.append('_unk_')
        self.sym_idx['_unk_'] = len(self.symbols)-1

    
        count = np.zeros((self.N,len(self.symbols)))

        for i,smile in enumerate(self.smiles):
            c = Counter(smile)
            c = {self.sym_idx[k]:v for k,v in c.items()}
            for k,v in c.items():
                count[i][k] = v
                
        return count

    def transform(self, smiles):
        count = np.zeros((len(smiles),len(self.symbols)))

        for i,smi in enumerate(smiles):
            for char in smi:
                if char in self.sym_idx.keys():
                    count[i][self.sym_idx[char]]+=1
                else:
                    count[i][self.sym_idx['_unk_']]+=1

        return count

class VectorRepresentation(CreateSymbolDataset):
    def __init__(self,smiles):
        super(VectorRepresentation, self).__init__(smiles)
    
    def fit(self, max_len = None):
        self._get_list_symbols()
        self.symbols.append('_unk_')
        self.sym_idx['_unk_'] = len(self.symbols)-1
        self.symbols.append('_pad_')
        self.sym_idx['_pad_'] = len(self.symbols)-1

        if max_len is None or max_len < self.max_size:
            max_len = self.max_size

        self.max_len = max_len

        res = []
        for smile in self.smiles:
            temp = np.array(list(map(lambda x: self.sym_idx[x], list(smile))))
            temp = np.pad(temp, (0,self.max_len - len(temp)), mode='constant',constant_values=(self.sym_idx['_pad_']))
            res.append(temp)

        return np.array(res)

    def _smile_to_idx(self,     smile):
        vec = []
        for x in list(smile):
            if x in self.sym_idx.keys():
                vec.append(self.sym_idx[x])
            else:
                vec.append(self.sym_idx['_unk_'])
                
        vec = np.array(vec)
        if len(vec) > self.max_len:
            vec = vec[:self.max_len]
        elif len(vec) <= self.max_len:
            vec = np.pad(vec, (0,self.max_len - len(vec)), mode='constant',constant_values=(self.sym_idx['_pad_']))
                
        return vec

    def transform(self, smiles):
        temp = np.array(list(map(lambda x: self._smile_to_idx(x), smiles)))
        return temp

class MorganFingerprints(CreateSymbolDataset):
    def __init__(self, smiles):
        super(MorganFingerprints, self).__init__(smiles)
        self.fingerprint_array = []

    def get_bit_vector_from_smile(self, smile, radius):

        mol = Chem.MolFromSmiles(smile)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol,radius)
        self.fingerprint_array.append(fingerprint)
        return np.array(list(fingerprint.ToBitString()),int)

    def transform(self, radius = 2):
        result = []
        self.fingerprint_array = []
        for smile in self.smiles:
            fingerprint = self.get_bit_vector_from_smile(smile, radius)
            result.append(fingerprint)
        return  np.array(result)


class MurckoScaffold(CreateSymbolDataset):
    def __init__(self, smiles):
        super(MurckoScaffold, self).__init__(smiles)
        

    def get_scaffold(self, smile):
        return MS.MurckoScaffoldSmilesFromSmiles(smile)  

    def transform(self):
        scaffold_smiles = np.array(list(map(self.get_scaffold,self.smiles)))
        return scaffold_smiles