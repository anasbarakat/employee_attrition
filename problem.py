# -*- coding: utf-8 -*-
"""
@author: Employee Attrition group
"""

import pandas as pd
import random
import numpy as np
import rampwf as rw
import os
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score,precision_score,recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score

class BaseScoreType(object):
    def check_y_pred_dimensions(self, y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError(
                'Wrong y_pred dimensions: y_pred should have {} instances, '
                'instead it has {} instances'.format(len(y_true), len(y_pred)))

    @property
    def worst(self):
        if self.is_lower_the_better:
            return self.maximum
        else:
            return self.minimum

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        if valid_indexes is None:
            valid_indexes = slice(None, None, None)
        y_true = ground_truths.y_pred[valid_indexes]
        y_pred = predictions.y_pred[valid_indexes]
        self.check_y_pred_dimensions(y_true, y_pred)
        return self.__call__(y_true, y_pred)
    
class ClassifierBaseScoreType(BaseScoreType):
    def score_function(self, ground_truths, predictions, valid_indexes=None):
        self.label_names = ground_truths.label_names
        if valid_indexes is None:
            valid_indexes = slice(None, None, None)
        y_pred_label_index = predictions.y_pred_label_index[valid_indexes]
        y_true_label_index = ground_truths.y_pred_label_index[valid_indexes]
        self.check_y_pred_dimensions(y_true_label_index, y_pred_label_index)
        return self.__call__(y_true_label_index, y_pred_label_index)


class metric(ClassifierBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='MCC_score', precision=3):
        
        self.name = name
        self.precision = precision
    

    def __call__(self, y_true, y_pred):
        
        score = matthews_corrcoef(y_true, y_pred)
        return score

class cohen(ClassifierBaseScoreType):
	is_lower_the_better = True
	minimum = 0.0
	maximum = float('inf')

	def __init__(self, name='cohen_kappa_score', precision=3):

		self.name = name
		self.precision = precision


	def __call__(self, y_true, y_pred):
	    
	    score = cohen_kappa_score(y_true, y_pred)
	    return score


problem_title = ' Employee Attrition group'
_target_column_name = 'left'
_prediction_label_names = [0,1,2,3,4]

# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)

# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorClassifier()

#    rw.score_types.f1_score(name='F1 Score', precision=3),
#     rw.score_types.precision_score(name='Precision Score', precision=3),
#    rw.score_types.recall_score(name='Recall Score', precision=3) 
score_types = [metric(),
cohen(),
 rw.score_types.accuracy.Accuracy(name='Accuracy', precision=3)
]
                                  


def get_cv(X, y):
    ''' cross-validation using stratified 
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

