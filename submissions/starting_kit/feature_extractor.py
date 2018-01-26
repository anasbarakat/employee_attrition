# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 

class FeatureExtractor():

    def __init__(self):
        return self
    
    def dummify(df):
        dum_dep = pd.get_dummies(df['department'])
        df = df.drop('department', axis=1)
        df = df.join(dum_dep)
        #add dummy variable
        #df = data.join(datanew)
        dum_sal = pd.get_dummies(df['salary'])
        df = df.drop('salary', axis=1)
        df = df.join(dum_sa)
        #df = data.join(datanew)
        return (df)
    
    def fit(self, X_df, y=None):
        return self

    def fit_transform(self, X_df, y=None):
        return self.transform(X_df)

    def transform(self, X_df):
        X = self.dummify(X_df)
        return X


