################
# Author: Reza C Doobary
#
# preprocessing.py
#
# The script provides functionality for various preprocessing steps used.
################

# Imports
import numpy as np
import pandas as pd 
from collections import Counter

# A Try-Except block is added for this script is used in the google colab environment - there we do not want to import rdkit unnecessarily since it is 
# a non-trivial and best avoided if possible (i.e must use conda - cannot use standard pip)
try: 
  from rdkit.Chem.Scaffolds import MurckoScaffold as MS
  from rdkit import Chem
  from rdkit.Chem import AllChem, Descriptors
except ModuleNotFoundError:
  print('Continue if using colab and not planning to use rdkit')


def _split(sm:str)->list:
    '''
    This functions splits a smile, represented by a string into a list of atom components.

    Note : This was not created by the author, but was taken from elsewhere - pending reference URL.
    '''
    arr = []
    i = 0
    while i < len(sm)-1:
        if not sm[i] in ['%', 'C', 'B', 'S', 'N', 'R', 'X', 'L', 'A', 'M', \
                        'T', 'Z', 's', 't', 'H', '+', '-', 'K', 'F']:
            arr.append(sm[i])
            i += 1
        elif sm[i]=='%':  
            arr.append(sm[i:i+3])
            i += 3
        elif sm[i]=='C' and sm[i+1]=='l':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='C' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='C' and sm[i+1]=='u':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='r':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='B' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='S' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='S' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='S' and sm[i+1]=='r':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='N' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='N' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='R' and sm[i+1]=='b':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='R' and sm[i+1]=='a':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='X' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='L' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='l':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='s':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='g':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='A' and sm[i+1]=='u':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='M' and sm[i+1]=='g':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='M' and sm[i+1]=='n':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='T' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='Z' and sm[i+1]=='n':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='s' and sm[i+1]=='i':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='s' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='t' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='H' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='+' and sm[i+1]=='2':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='+' and sm[i+1]=='3':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='+' and sm[i+1]=='4':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='-' and sm[i+1]=='2':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='-' and sm[i+1]=='3':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='-' and sm[i+1]=='4':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='K' and sm[i+1]=='r':
            arr.append(sm[i:i+2])
            i += 2
        elif sm[i]=='F' and sm[i+1]=='e':
            arr.append(sm[i:i+2])
            i += 2
        else:
            arr.append(sm[i])
            i += 1
    if i == len(sm)-1:
        arr.append(sm[i])
    return ' '.join(arr)

class CreateSymbolDataset(object):
    """
    Base class responsibe for enumerating symbols (atoms) with the smile.
    """
    def __init__(self, smiles):
        self.smiles = smiles
        self.max_size = self._get_max_size()
        self.N = len(self.smiles)
        
    def _get_max_size(self)->int:
        """
        Compute the size of each smile in self.smiles and finds the maximum size.
        """
        max_size = max(map(len,self.smiles))
        return max_size
    
    def _get_list_symbols(self, splitter:str = None)->None:
        """
        Amagamates all smiles, finds the key set of symbols that make up all smiles
        and creates a list of symbols and a symbol-index dictionary.
        """
        symbols = set()
        for smile in self.smiles:
            if splitter:
                symbols = symbols.union(set(list(smile.split(splitter))))
            else:
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
    """
    Class responsible for creating a bag of words-like dataset based off by treating the characters of the smiles as individual symbols - thus,
    this does not consider atoms in the chemistry sense.

    Basic usage:

    >> smiles = X_train_source['SMILES'].values
    >> bow = pp.BagOfWords(smiles)
    >> bow_train = bow.fit()
    >> bow_test = bow.transform(X_test_source['SMILES'].values)
    """
    def __init__(self,smiles:list):
        super(BagOfWords, self).__init__(smiles)
    
    def fit(self)->np.array:
        """
        The main fit function, which counts the number of symbols within each smile and create a np.array of counts for each smile and symbol.
        """
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

    def transform(self, smiles:list)->np.array:
        """
        Transforms the smiles list subject to the fit function.
        """
        count = np.zeros((len(smiles),len(self.symbols)))

        for i,smi in enumerate(smiles):
            for char in smi:
                if char in self.sym_idx.keys():
                    count[i][self.sym_idx[char]]+=1
                else:
                    count[i][self.sym_idx['_unk_']]+=1

        return count

class BagOfWordsMols(CreateSymbolDataset):
    """
    Class responsible for creating a bag of words-like dataset based off by symbols which are understood to be atoms by making use of the _split function.

    Basic usage:

    >> smiles = X_train_source['SMILES'].values
    >> bow = pp.BagOfWordsMols(smiles)
    >> bow_train = bow.fit()
    >> bow_test = bow.transform(X_test_source['SMILES'].values)
    """
    def __init__(self, smiles:list):
        super(BagOfWordsMols, self).__init__(smiles)
        # takes the smiles and applies the _split function to each smile in the list labelled smiles.
        self.smiles = self._isolate_mols_from_smiles(smiles)
        

    def _isolate_mols_from_smiles(self, smiles:list)->np.array:
        """
        Applies the splitter to all smiles in the list smiles.
        """
        smiles = np.array([_split(smi) for smi in smiles])
        return smiles
        
    def fit(self):
        """
        The main fitting function - importantly and in distinction to the generic case, the list of symbols now has a splitting string ' ', since 
        this has come from the use of the _split function.
        """
        self._get_list_symbols(' ')
        self.symbols.append('_unk_')
        self.sym_idx['_unk_'] = len(self.symbols)-1
        
        count = np.zeros((self.N,len(self.symbols)))
        
        for i,smile in enumerate(self.smiles):
            c = Counter(smile.split(' '))
            c = {self.sym_idx[k]:v for k,v in c.items()}
            for k,v in c.items():
                count[i][k] = v
        
        return count
    
    def transform(self, smiles:list)->np.array:
        count = np.zeros((len(smiles),len(self.symbols)))

        for i,smi in enumerate(smiles):
            for char in smi:
                if char in self.sym_idx.keys():
                    count[i][self.sym_idx[char]]+=1
                else:
                    count[i][self.sym_idx['_unk_']]+=1

        return count
        


class MorganFingerprints(CreateSymbolDataset):
    """
    Class that finds the morgan fingerprint for the smiles.

    Basic usage:

    >> smiles = data['SMILES'].values
    >> mf = MorganFingerprints(smiles)
    >> X = mf.transform(radius)
    """
    def __init__(self, smiles:list):
        super(MorganFingerprints, self).__init__(smiles)
        self.fingerprint_array = []

    def get_bit_vector_from_smile(self, smile:str, radius:int)->np.array:
        """
        Given a smile and a radius, rdkit is employed for the purpose of finding the fingerprint representation of the smile.
        """
        mol = Chem.MolFromSmiles(smile)
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol,radius)
        self.fingerprint_array.append(fingerprint)
        return np.array(list(fingerprint.ToBitString()),int)

    def transform(self, radius:int = 2)->np.array:
        """
        Implements the previous function self.get_bit_vector_from_smile for all of the smiles.
        """
        result = []
        self.fingerprint_array = []
        for smile in self.smiles:
            fingerprint = self.get_bit_vector_from_smile(smile, radius)
            result.append(fingerprint)
        return  np.array(result)


class MurckoScaffold(CreateSymbolDataset):
    """
    This class finds the scaffolding for a given smiles, and applies it to a list of smiles - it is essentially a wrapper around the core
    functionality of rdkit.

    Basic usage:

    >> smiles = data['SMILES'].values
    >> ms = pp.MurckoScaffold(smiles)
    >> X = ms.transform()
    """
    def __init__(self, smiles:list):
        super(MurckoScaffold, self).__init__(smiles)
        
    def get_scaffold(self, smile:str)->str:
        return MS.MurckoScaffoldSmilesFromSmiles(smile)  

    def transform(self)->list:
        scaffold_smiles = np.array(list(map(self.get_scaffold,self.smiles)))
        return scaffold_smiles


# the following is a list of molecular descriptors which are used by rdkit.
all_descriptors = ['Asphericity',
 'Eccentricity',
 'InertialShapeFactor',
 'NPR1',
 'NPR2',
 'PMI1',
 'PMI2',
 'PMI3',
 'RadiusOfGyration',
 'SpherocityIndex',
 'MaxEStateIndex',
 'MinEStateIndex',
 'MaxAbsEStateIndex',
 'MinAbsEStateIndex',
 'qed',
 'MolWt',
 'HeavyAtomMolWt',
 'ExactMolWt',
 'NumValenceElectrons',
 'NumRadicalElectrons',
 'MaxPartialCharge',
 'MinPartialCharge',
 'MaxAbsPartialCharge',
 'MinAbsPartialCharge',
 'FpDensityMorgan1',
 'FpDensityMorgan2',
 'FpDensityMorgan3',
 'BalabanJ',
 'BertzCT',
 'Chi0',
 'Chi0n',
 'Chi0v',
 'Chi1',
 'Chi1n',
 'Chi1v',
 'Chi2n',
 'Chi2v',
 'Chi3n',
 'Chi3v',
 'Chi4n',
 'Chi4v',
 'HallKierAlpha',
 'Ipc',
 'Kappa1',
 'Kappa2',
 'Kappa3',
 'LabuteASA',
 'PEOE_VSA1',
 'PEOE_VSA10',
 'PEOE_VSA11',
 'PEOE_VSA12',
 'PEOE_VSA13',
 'PEOE_VSA14',
 'PEOE_VSA2',
 'PEOE_VSA3',
 'PEOE_VSA4',
 'PEOE_VSA5',
 'PEOE_VSA6',
 'PEOE_VSA7',
 'PEOE_VSA8',
 'PEOE_VSA9',
 'SMR_VSA1',
 'SMR_VSA10',
 'SMR_VSA2',
 'SMR_VSA3',
 'SMR_VSA4',
 'SMR_VSA5',
 'SMR_VSA6',
 'SMR_VSA7',
 'SMR_VSA8',
 'SMR_VSA9',
 'SlogP_VSA1',
 'SlogP_VSA10',
 'SlogP_VSA11',
 'SlogP_VSA12',
 'SlogP_VSA2',
 'SlogP_VSA3',
 'SlogP_VSA4',
 'SlogP_VSA5',
 'SlogP_VSA6',
 'SlogP_VSA7',
 'SlogP_VSA8',
 'SlogP_VSA9',
 'TPSA',
 'EState_VSA1',
 'EState_VSA10',
 'EState_VSA11',
 'EState_VSA2',
 'EState_VSA3',
 'EState_VSA4',
 'EState_VSA5',
 'EState_VSA6',
 'EState_VSA7',
 'EState_VSA8',
 'EState_VSA9',
 'VSA_EState1',
 'VSA_EState10',
 'VSA_EState2',
 'VSA_EState3',
 'VSA_EState4',
 'VSA_EState5',
 'VSA_EState6',
 'VSA_EState7',
 'VSA_EState8',
 'VSA_EState9',
 'FractionCSP3',
 'HeavyAtomCount',
 'NHOHCount',
 'NOCount',
 'NumAliphaticCarbocycles',
 'NumAliphaticHeterocycles',
 'NumAliphaticRings',
 'NumAromaticCarbocycles',
 'NumAromaticHeterocycles',
 'NumAromaticRings',
 'NumHAcceptors',
 'NumHDonors',
 'NumHeteroatoms',
 'NumRotatableBonds',
 'NumSaturatedCarbocycles',
 'NumSaturatedHeterocycles',
 'NumSaturatedRings',
 'RingCount',
 'MolLogP',
 'MolMR',
 'fr_Al_COO',
 'fr_Al_OH',
 'fr_Al_OH_noTert',
 'fr_ArN',
 'fr_Ar_COO',
 'fr_Ar_N',
 'fr_Ar_NH',
 'fr_Ar_OH',
 'fr_COO',
 'fr_COO2',
 'fr_C_O',
 'fr_C_O_noCOO',
 'fr_C_S',
 'fr_HOCCN',
 'fr_Imine',
 'fr_NH0',
 'fr_NH1',
 'fr_NH2',
 'fr_N_O',
 'fr_Ndealkylation1',
 'fr_Ndealkylation2',
 'fr_Nhpyrrole',
 'fr_SH',
 'fr_aldehyde',
 'fr_alkyl_carbamate',
 'fr_alkyl_halide',
 'fr_allylic_oxid',
 'fr_amide',
 'fr_amidine',
 'fr_aniline',
 'fr_aryl_methyl',
 'fr_azide',
 'fr_azo',
 'fr_barbitur',
 'fr_benzene',
 'fr_benzodiazepine',
 'fr_bicyclic',
 'fr_diazo',
 'fr_dihydropyridine',
 'fr_epoxide',
 'fr_ester',
 'fr_ether',
 'fr_furan',
 'fr_guanido',
 'fr_halogen',
 'fr_hdrzine',
 'fr_hdrzone',
 'fr_imidazole',
 'fr_imide',
 'fr_isocyan',
 'fr_isothiocyan',
 'fr_ketone',
 'fr_ketone_Topliss',
 'fr_lactam',
 'fr_lactone',
 'fr_methoxy',
 'fr_morpholine',
 'fr_nitrile',
 'fr_nitro',
 'fr_nitro_arom',
 'fr_nitro_arom_nonortho',
 'fr_nitroso',
 'fr_oxazole',
 'fr_oxime',
 'fr_para_hydroxylation',
 'fr_phenol',
 'fr_phenol_noOrthoHbond',
 'fr_phos_acid',
 'fr_phos_ester',
 'fr_piperdine',
 'fr_piperzine',
 'fr_priamide',
 'fr_prisulfonamd',
 'fr_pyridine',
 'fr_quatN',
 'fr_sulfide',
 'fr_sulfonamd',
 'fr_sulfone',
 'fr_term_acetylene',
 'fr_tetrazole',
 'fr_thiazole',
 'fr_thiocyan',
 'fr_thiophene',
 'fr_unbrch_alkane',
 'fr_urea']

class MolecularDescriptors:
    """
    Class responsible for find the molecular descriptors of the smiles.

    Basic usage:

    >> mol = Chem.MolFromSmiles(smile) 
    >> moldes = pp.MolecularDescriptors(mol)
    >> res = moldes.compute_all(mol)

    """
    def __init__(self, smiles:list):
        self.smiles = smiles
        self.descriptors3d = ['Asphericity',
         'Eccentricity',
         'InertialShapeFactor',
         'NPR1',
         'NPR2',
         'PMI1',
         'PMI2',
         'PMI3',
         'RadiusOfGyration',
         'SpherocityIndex']
        self.descriptors = Descriptors._descList
        
    def _get_3d_descriptors(self, mol)->dict:
        result = {}
        for func in self.descriptors3d:
            try:
                value = eval('Descriptors3D.'+func+'(mol)')
            except:
                value = np.nan
            result[func] = value
        return result
    
    def _get_descriptors(self, mol)->dict:
        result = {}
        for name, func in self.descriptors:
            try:
                value = func(mol)
            except:
                value = np.nan
            result[name] = value
        return result
    
    def compute_all(self, mol)->list:
        result = self._get_3d_descriptors(mol)
        result.update(self._get_descriptors(mol))
        return result


####### I will come back to commenting these functions - but they are under current developement and may change in due course.

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


class VectorRepresentationMols(CreateSymbolDataset):
    def __init__(self,smiles):
        super(VectorRepresentationMols, self).__init__(smiles)
        self.smiles = self._isolate_mols_from_smiles(smiles)

    def _isolate_mols_from_smiles(self, smiles):
        smiles = np.array([_split(smi) for smi in smiles])
        return smiles
    
    def fit(self, max_len = None):
        self._get_list_symbols(' ')
        self.symbols.append('_unk_')
        self.sym_idx['_unk_'] = len(self.symbols)-1
        self.symbols.append('_pad_')
        self.sym_idx['_pad_'] = len(self.symbols)-1

        if max_len is None or max_len < self.max_size:
            max_len = self.max_size

        self.max_len = max_len

        res = []
        for smile in self.smiles:
            temp = np.array(list(map(lambda x: self.sym_idx[x], smile.split(' '))))
            temp = np.pad(temp, (0,self.max_len - len(temp)), mode='constant',constant_values=(self.sym_idx['_pad_']))
            res.append(temp)

        return np.array(res)

    def _smile_to_idx(self, smile):
        vec = []
        for x in smile.split(' '):
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
        smiles = self._isolate_mols_from_smiles(smiles)
        temp = np.array(list(map(lambda x: self._smile_to_idx(x), smiles)))
        return temp
        
