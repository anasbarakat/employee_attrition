# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class Classifier(BaseEstimator):
    def __init__(self):
        #self.clf = RandomForestClassifier()
        self.clf = LogisticRegression()

    def fit(self, X, y):
        #self.clf.fit(X.todense(), y)
        self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
