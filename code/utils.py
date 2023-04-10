"""
SUPPLEMENTARY CODE FOR
Title: Using Machine Learning to predict Financial Crises: An Evaluation of different Learning Algorithms for Early Warning Models.
Author: Chris Reimann.

This file implements custom data splits used in the experiments (InSample, CorssValidation, StrictForecast).
"""

from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

class CustomGroupKFold:
    def __init__(self, n_splits = 5):
        self.n_splits = n_splits
    
    def get_n_splits(self, x = None, y = None, groups = None):
        return self.n_splits
    
    def split(self, data, y = None, groups = None):
        kf = KFold(n_splits = self.n_splits, shuffle = True)
        base = list(kf.split(data, y = y, groups = groups))
        foldsNew = []
        
        if type(data) == np.ndarray:
            data = pd.DataFrame(data).rename(columns={0:"iso", 1:"year", 2:"crisisID", 3:"crisisRisk"})
        
        for fold in base:
            ixTrain, ixTest = fold
            dataTest = data.loc[ixTest]
            crisisIDsTest = dataTest[dataTest.crisisRisk == 1].crisisID.unique()
            ixChange = data[data.crisisID.isin(crisisIDsTest)].index
            
            new_ixTrain = np.setdiff1d(ixTrain, ixChange)
            new_ixTest = np.union1d(ixTest, ixChange)
            foldsNew.append((new_ixTrain, new_ixTest))
        
        return foldsNew
    
    
class CustomTimeSeriesSplit:
    def __init__(self, n_splits = 5):
        self.n_splits = n_splits
    
    def get_n_splits(self, x = None, y = None, groups = None):
        return self.n_splits
    
    def split(self, data, y = None, groups = None):
        ts = TimeSeriesSplit(n_splits = self.n_splits)
        base = list(ts.split(data, y = y, groups = groups))
        foldsNew = []
        
        if type(data) == np.ndarray:
            data = pd.DataFrame(data).rename(columns={0:"iso", 1:"year", 2:"crisisID", 3:"crisisRisk"})
        
        for fold in base:
            ixTrain, ixTest = fold
            dataTest = data.loc[ixTest]
            
            crisisIDsTest = dataTest[dataTest.crisisRisk == 1].crisisID.unique()
            ixChange = data[data.crisisID.isin(crisisIDsTest)].index

            new_ixTrain = np.setdiff1d(ixTrain, ixChange)
            new_ixTest = np.union1d(ixTest, ixChange)
            foldsNew.append((new_ixTrain, new_ixTest))
            self.n_splits = len(foldsNew)
        return foldsNew
    
    
class InSampleSplit():
    def __init__(self):
        self.n_splits = 1
        
    def get_n_splits(self, x = None, y = None, groups = None):
        return self.n_splits
    
    def split(self, data, y = None, groups = None):
        if type(data) == np.ndarray:
            data = pd.DataFrame(data)
        ix = data.index.tolist()
        return [[ix, ix]]
    

    