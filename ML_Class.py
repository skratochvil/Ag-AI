# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:54:45 2020

@author: Donovan
"""
class Active_ML_Model:
    def getLabels(image_names):
        """
        Sends a list of images for the user to label.
        retrieves the user's answer
        returns labels
        """
        return None
    
    def __init__(self, data, ml_classifier, preprocess, n_samples = 5):
        from sklearn.utils import shuffle
        data = shuffle(data)                                                    #Shuffle the Data
        X_train = data.iloc[:n_samples, :]   
        self.X_test = data.iloc[n_samples:, :]                                   #First n_samples Rows
        labels = getLabels(list(X_train.index))
        X_train['y_value'] = labels
        self.X_train = X_train
        self.ml_model = ML_Model(X_train, ml_classifier, preprocess)
    
    def Continue(self, n_samples = 5, sampling_method):
        X_sample, self.X_test = sampling_method(self.ml_model, n_samples)
        labels = getLables(list(X_sample.index))
        X_sample['y_value'] = labels
        self.X_train = pd.concat([self.X_train, X_sample])
        self.ml_model = ML_Model(X_train, ml_classifier, preprocess)
    
    def sendProgress(self):
        return None
    
    def sendResults(self):
        return None

class ML_Model:
    """
    This class creates a machine learning model based on the data sent, data preprocessing, and type of ml classifier.
    
    
    """
            
    def __init__(self, train_data, ml_classifier, preprocess):
        self.ml_classifier = ml_classifier
        self.preprocess = preprocess
        
#       Split data X and y(resulting variable)
#       This will most likely change after we decide how to store the data
        self.X = train_data.iloc[:,: -1].values
        self.y = train_data.iloc[:, -1].values

        self.X = self.preprocess.fit_transform(self.X)
        
#       Build Model
        self.ml_model = ml_classifier.fit(self.X, self.y)
        
    def GetKnownPredictions(self, new_data):
        """
        
        """
        new_data_X = new_data.iloc[:, :-1].values
        new_data_X = self.preprocess.transform(new_data_X)
        y_prediction = self.ml_model.predict(new_data_X)
        y_probabilities = self.ml_model.predict_proba(new_data_X)
        y_probabilities = [max(prob) for prob in y_probabilities]
        return y_prediction, max(y_probabilities)
    
    def GetUnknownPredictions(self, new_data_X):
        new_data_X = self.preprocess.transform(new_data_X)
        y_prediction = self.ml_model.predict(new_data_X)
        y_probabilities = self.ml_model.predict_proba(new_data_X)
        y_probabilities = [max(prob) for prob in y_probabilities]
        return y_prediction, y_probabilities
    
    def K_fold(self):
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(self.ml_classifier, self.X, self.y, cv=10)    
        return accuracies