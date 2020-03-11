# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:48:17 2020

@author: Donovan
"""

def lowestPercentage(al_model, n):
    from sklearn.utils import shuffle
    X_test = al_model.X_test
    predictions, probabilities = al_model.ml_model.GetUnknownPredictions(X_test)

    X_test['prediction score'] = probabilities
    X_test.sort_values('prediction score', axis = 1)
    
    X_sample = X_test.iloc[:n, :]
    new_X_test = X_test.iloc[n:, :]
    return shuffle(X_sample), shuffle(new_X_test.shuffle())