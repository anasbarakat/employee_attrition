# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np 


def add_missing_dummies( X_df, cols ):
    missing_dums = set( cols ) - set( X_df.columns )
    for i in missing_dums:
        X_df[i] = 0

def final_dummies_fix( X_df, cols ):  

    add_missing_dummies( X_df, cols )
# Make sure all columns exists after the add of missing dummies
    try:
        assert( set( cols ) - set( X_df.columns ) == set())
    except AssertionError:
    	print("you haven't added all the missing dummies !!!")

    #print ( "perfect add !!")
# Here we check if the  columns in the test and the train samples are the same.
    ext_columns = set( X_df.columns ) - set( cols )
    if ext_columns:
        print ("there are extra columns in your features:", ext_columns)
# Organizing features and make sure that the order of data in the train is always the same 
    X_df = X_df[ cols ]
    return X_df



def dummify( df):
	""" This function is used to generate dummy variables from some features """
	# dummy of department
	dum_dep = pd.get_dummies(df['department'])
	df = df.drop('name', axis=1)
	df = df.drop('department', axis=1)
	df = df.join(dum_dep)
	
	# dummy of salary
	dum_sal = pd.get_dummies(df['salary'])
	df = df.drop('salary', axis=1)
	df = df.join(dum_sal)
	return (df)



class FeatureExtractor():

    def __init__(self):
        self.columns=['Unnamed: 0', 'satisfaction_level', 'last_evaluation','number_projects', \
                      'average_monthly_hours', 'time_spent_company','work_accident', 'promotion_last_5_years', \
                      'salary_level', 'IT','RandD', 'accounting', 'hr', 'management', 'marketing', 'product_mng','sales',\
                      'support', 'technical', 'high', 'low', 'medium']

    def fit(self, X_df, y=None):
    	return X_df

    def fit_transform(self, X_df, y=None):
        return self.transform(X_df)

    def transform(self, X_df):
        X = dummify(X_df)
        return final_dummies_fix(X,self.columns)