# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:54:45 2020

@author: Donovan
"""

class ML_Model:
    def __init__(self, data, ml_classifier, DataPreprocessing = None):
        self.data = data
        
#       Split data X and y(resulting variable)
#       This will most likely change after we decide how to store the data
        self.X = data.iloc[:, : -1].values
        self.y = data.iloc[:, -1].values

#       Split data into train and test set
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.25, random_state = 0)

#       Preprocess X
        if DataPreprocessing != None:
            self.X_train_processed, self.X_test_processed = DataPreprocessing(self.X_train, self.X_test)
        else:
            self.X_train_processed, self.X_test_processed = self.X_train, self.X_test
        
        
        self.ml_model = ml_classifier.fit(self.X_train_processed, self.y_train)
        self.y_pred = self.ml_model.predict(self.X_test_processed)