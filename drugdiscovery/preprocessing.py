import numpy as np
import pandas as pd 
from collections import Counter
from rdkit.Chem.Scaffolds import MurckoScaffold as MS
from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit
from rdkit.Chem import Descriptors, Descriptors3D

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

import numpy as np
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
    def __init__(self, smiles):
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
        
    def _get_3d_descriptors(self, mol):
        result = {}
        for func in self.descriptors3d:
            try:
                value = eval('Descriptors3D.'+func+'(mol)')
            except:
                value = np.nan
            result[func] = value
        return result
    
    def _get_descriptors(self, mol):
        result = {}
        for name, func in self.descriptors:
            try:
                value = func(mol)
            except:
                value = np.nan
            result[name] = value
        return result
    
    def compute_all(self, mol):
        result = self._get_3d_descriptors(mol)
        result.update(self._get_descriptors(mol))
        return result