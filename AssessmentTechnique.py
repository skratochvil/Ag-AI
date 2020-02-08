# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:23:56 2020

@author: Donovan
"""

def K_fold(ml_classifier):
    from sklearn.model_selection import cross_val_score

    accuracies = cross_val_score(ml_classifier, ml_classifier.X, ml_classifier.y, cv=10)    
    return accuracies