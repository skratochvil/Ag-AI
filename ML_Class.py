# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:54:45 2020

@author: Donovan
"""

class ML_Model:
            
    def __init__(self, train_data, ml_classifier, DataPreprocessing = None):
        self.DataPreprocessing = DataPreprocessing
        self.ml_classifier = ml_classifier
        
#       Split data X and y(resulting variable)
#       This will most likely change after we decide how to store the data
        self.X = train_data.iloc[:, : -1].values
        self.y = train_data.iloc[:, -1].values

#       Preprocess X
        if self.DataPreprocessing != None:
            self.X, self.preprocess_technique = self.DataPreprocessing(self.X)
        else:
            self.preprocess_technique = None
        
#       Build Model
        self.ml_model = ml_classifier.fit(self.X, self.y)
        
    def GetKnownPredictions(self, new_data):
        new_data_X = new_data.iloc[:, :-1].values
        if self.preprocess_technique != None:
            new_data_X = self.preprocess_technique.fit(new_data_X)
        y_prediction = self.ml_model.predict(new_data_X)
        return y_prediction
    
    def GetUnknownPredictions(self, new_data_X):
        if self.preprocess_technique != None:
            new_data_X = self.preprocess_technique.fit(new_data_X)
        y_prediction = self.ml_model.predict(new_data_X)
        return y_prediction

    def K_fold(self):
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(self.ml_classifier, self.X, self.y, cv=10)    
        return accuracies