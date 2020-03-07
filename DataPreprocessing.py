# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:21:32 2020

@author: Donovan
"""
class DataPreprocessing:
    def __init__(self, standard_scaling = False, normalization = False, pca = False, components = None):
        self.sc = None
        if standard_scaling == True:
            from sklearn.preprocessing import StandardScaler
            self.sc = StandardScaler()
        
        self.norm = None
        if normalization == True:
            from sklearn.preprocessing import Normalizer
            self.norm = Normalizer()
        
        self.pca = None
        if pca == True:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components = components)

        
    def fit_transform(self, X_train):
        if self.sc != None:
            X_train = self.sc.fit_transform(X_train)
        if self.norm != None:
            X_train = self.norm.fit_transform(X_train)
        if self.pca != None:
            X_train = self.pca.fit_transform(X_train)
        return X_train
    
    def transform(self, X_test):
        if self.sc != None:
            X_test = self.sc.transform(X_test)
        if self.norm != None:
            X_test = self.norm.transform(X_test)
        if self.pca != None:
            X_test = self.pca.transform(X_test)
        return X_test