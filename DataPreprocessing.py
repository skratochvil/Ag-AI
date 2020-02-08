# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:21:32 2020

@author: Donovan
"""
def StandardScaling(X_train):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    return X_train, sc