################
# Author: Reza C Doobary
#
# dataintegrity.py
#
# The purpose of the script is provide functionality for an initial clean of the data received from the Tox-21 dataset challenge.
################


# Imports
import numpy as np 
import pandas as pd
import re
from functools import reduce
from tqdm import tqdm

# A Try-Except block is added for this script is used in the google colab environment - there we do not want to import rdkit unnecessarily since it is 
# a non-trivial and best avoided if possible (i.e must use conda - cannot use standard pip)
try: 
 from rdkit.Chem import PandasTools
except ModuleNotFoundError:
  print('Continue if using colab and not planning to use rdkit')

class DataIntegrity:
    """
    Class responsible for performing some basic cleaning to the smiles .sdf dataset.

    Basic usage:

    >> filename = 'data/tox21_10k_data_all.sdf'
    >> di = DataIntegrity(filename)
    >> di.clean_columns(['FW'])
    >> target_columns = ['SR-HSE',
        'NR-AR', 'SR-ARE', 'NR-Aromatase', 'NR-ER-LBD', 'NR-AhR', 'SR-MMP',
        'NR-ER', 'NR-PPAR-gamma', 'SR-p53', 'SR-ATAD5', 'NR-AR-LBD']
    >> di.change_types(target_columns, float)
    >> di.merge_duplicate_target_rows('SMILES',target_columns)
    >> di.save('data/data_dups_removed.sdf','data/data_dups_removed.csv')
    """
    def __init__(self, filename:str):
        self.source = PandasTools.LoadSDF(filename,smilesName='SMILES',
                                    molColName='Molecule',
                                    includeFingerprints=True)

    
    def _clean_number(self, string:str)->float:
        """
        Splits a string with '(' and ' ', and returns the first element of the split 
        string recasted as a float.
        """
        res = re.split(r'[(\s]\s*', string)  
        return  float(res[0])

    def _adding(self, x:float, y:float)->float:
        """
        Performs 'adding' in the following way:

        _adding(nan,y) = y
        _adding(x,nan) = x
        _adding(x,y) where x==y = x
        _adding(1,0) = 1
        _adding(0,1) = 1
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

    def _adding_pd(self, series:pd.core.series.Series)->pd.core.series.Series:
        """
        Takes a pandas series object and performs the summation as per self._adding.
        """
        return reduce(lambda x, y: self._adding(x,y), series)

    def _find_duplicates(self, column_name:str)->np.array:
        """
        Finds all the duplicated rows for the given column_name is the source file.
        """
        duplicated_smiles = self.source[column_name][self.source[column_name].duplicated()]
        return duplicated_smiles.values

    def clean_columns(self, column_names:list)->None:
        """
        Cleans a column in the source file according to self._clean_number.
        """
        for c_names in column_names:
            self.source[c_names] = self.source[c_names].apply(lambda x: self._clean_number(x))
    
    def change_types(self, column_names:list, type)->None:
        """
        Type recasts a column in the source file.
        """
        self.source = self.source.astype({t:type for t in column_names})

    def merge_duplicate_target_rows(self, duplicate_column:str, target_columns:list)->None:
        """
        This is the main functionality of the class and somewhat specific to the use case in hand.
        We have a dataset with smiles and targets for which the activity of the candidate molecule is assessed.
        However, over time (groups - experiement chemistry groups) have added to the dataset rather than modifying 
        old entries, thus previously untested molecules have becomes tested as the row number increases. This leads to 
        duplicate smiles in the dataset and target rows which only partially caputre the target information.
        
        
        - this function finds the duplicated rows and subject to the _adding merges the rows together.
        """
        # step 1. Find the duplicates.
        dups = self._find_duplicates(duplicate_column)

        # For each duplicate found :
        for d in tqdm(dups):
            
            # step 2. For those targets for which duplicates exists amalgamate the data using the _adding function.
            temp = self.source[self.source[duplicate_column] == d][target_columns].apply(self._adding_pd)
            # step 3. Keep track of the indices of the duplicates.
            indx = list(self.source[self.source[duplicate_column] == d].index)

            # step 4. Take note that we keep the first index and throw away the remaining indices.
            to_keep = indx[0]
            to_throw = indx[1:]

            # step 5. Find the row that we want to keep, update it with the amalgamated data.
            tmp2 = self.source.loc[to_keep]
            tmp2.update(temp)
            self.source.loc[to_keep] = tmp2

            # step 6. Drop the unwanted rows.
            for del_idx in to_throw:
                self.source = self.source.drop(index = del_idx)


    def save(self, filename_sdf:str, filename_csv:str)->None:
        """
        Saves the results file as both and sdf and a csv. 
        Both are used since the sdf file takes longer to load than a simple csv file.
        """
        PandasTools.WriteSDF(self.source, filename_sdf, \
            molColName='Molecule', properties=list(self.source.columns))
        self.source.to_csv(filename_csv)
        