# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:21:32 2020

@author: Donovan
"""
def StandardScaling(X_train, X_test):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test