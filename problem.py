# -*- coding: utf-8 -*-
"""
@author: Employee Attrition group
"""

import pandas as pd
import random
import numpy as np
import rampwf as rw
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score,precision_score,recall_score
from sklearn.model_selection import StratifiedShuffleSplit

problem_title = ' Employee Attrition group'
_target_column_name = 'left'
_prediction_label_names = [0, 1]

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()


score_types = [
    rw.score_types.roc_auc(name='ROC-AUC', precision=3),
    rw.score_types.f1_score(name='F1 Score', precision=3),
    rw.score_types.precision_score(name='Precision Score', precision=3),
    rw.score_types.recall_score(name='Recall Score', precision=3)                                   
]


def get_cv(X, y):
    ''' cross-validation using stratfied 
    '''
    cv = StratifiedShuffleSplit(n_splits= 5, test_size= 0.5, random_state=43)

    return cv.split(X, y)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name), sep=';')
    y_array = data[_target_column_name].values
    X_df = data.drop(_target_column_name, axis=1)
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        return X_df[:100], y_array[:100]
    else:
        return X_df, y_array
    

def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
    