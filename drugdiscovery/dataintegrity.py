import numpy as np 
import pandas as pd
import re
from functools import reduce
from tqdm import tqdm

from rdkit.Chem import PandasTools

class DataIntegrity:
    def __init__(self, filename):
        self.source = PandasTools.LoadSDF(filename,smilesName='SMILES',
                                    molColName='Molecule',
                                    includeFingerprints=True)

    
    def _clean_number(self, string):
        """
        split a string with '(' and ' '
        """
        res = re.split(r'[(\s]\s*', string)  
        return  float(res[0])

    def _adding(self, x, y):
        """
        Function that knows how to add given that one or both of the numbers are nan
        """
        if np.isnan(y) and not np.isnan(x):
            return x
        elif not np.isnan(y) and np.isnan(x):
            return y
        elif np.isnan(y) and np.isnan(x):
            return y
        elif not np.isnan(y) and not np.isnan(x):
            if y==x:
                return y
            elif y==1.0 and x==0.0:
                return y
            elif x == 1.0 and y == 0.0:
                return x

    def _adding_pd(self, series):
       return reduce(lambda x, y: self._adding(x,y), series)

    def _find_duplicates(self, column_name):
        duplicated_smiles = self.source[column_name][self.source[column_name].duplicated()]
        return duplicated_smiles.values

    def clean_columns(self, column_names):
        for c_names in column_names:
            self.source[c_names] = self.source[c_names].apply(lambda x: self._clean_number(x))
    
    def change_types(self, column_names, type):
        self.source = self.source.astype({t:float for t in column_names})

    def merge_duplicate_target_rows(self, duplicate_column, target_columns):
        dups = self._find_duplicates(duplicate_column)

        for d in tqdm(dups):
            
            temp = self.source[self.source[duplicate_column] == d][target_columns].apply(self._adding_pd)
            indx = list(self.source[self.source[duplicate_column] == d].index)

            to_keep = indx[0]
            to_throw = indx[1:]

            tmp2 = self.source.loc[to_keep]
            tmp2.update(temp)
            self.source.loc[to_keep] = tmp2

            for del_idx in to_throw:
                self.source = self.source.drop(index = del_idx)


    def save(self, filename_sdf, filename_csv):
        PandasTools.WriteSDF(self.source, filename_sdf, \
            molColName='Molecule', properties=list(self.source.columns))
        self.source.to_csv(filename_csv)
        