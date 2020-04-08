# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:54:45 2020

@author: Donovan
"""
class ML_Model:
    """
    This class creates a machine learning model based on the data sent, 
    data preprocessing, and type of ml classifier.
    
    """

    def __init__(self, train_data, ml_classifier, preprocess):
        """
        This function controls the initial creation of the machine learning model.
        
        Parameters
        ----------
        train_data : pandas DataFrame
            The data the machine learning model will be built on.
        ml_classifier : classifier object
            The classifier to be used to create the machine learning model.
        preprocess : Python Function
            The function used to preprocess the data before model creation.
            
        Attributes
        -------
        ml_classifier : classifier object
            The classifier to be used to create the machine learning model.
        preprocess : Python Function
            The function used to preprocess the data before model creation.
        X : pandas DataFrame
            The features in the train set.
        y : pandas Series
            The responce variable.
        ml_model : fitted machine learning classifier
            The machine learning model created using the training data.
        """
        self.ml_classifier = ml_classifier
        self.preprocess = preprocess
        
        self.X = train_data.iloc[:,: -1].values
        self.y = train_data.iloc[:, -1].values

        self.X = self.preprocess.fit_transform(self.X)
        
        self.ml_model = ml_classifier.fit(self.X, self.y)
        
    def GetKnownPredictions(self, new_data):
        """
        This function predicts the labels for a new set of data that contains labels. 
        It returns these predictions and the probability. 
        
        Parameters
        ----------
        new_data : pandas DataFrame
            The new data to be labeled.
            
        Returns
        -------
        y_prediction : list
            list of predicted labels.
        prob : list
            The probability that the label is correct.
        """
        new_data_X = new_data.iloc[:, :-1].values
        new_data_X = self.preprocess.transform(new_data_X)
        y_prediction = self.ml_model.predict(new_data_X)
        y_probabilities = self.ml_model.predict_proba(new_data_X)
        y_probabilities = [max(prob) for prob in y_probabilities]
        return y_prediction, max(y_probabilities)
    
    def GetUnknownPredictions(self, new_data_X):
        """
        This function predicts the labels for a new set of data that does not contains labels. 
        It returns these predictions and the probability. 
        
        Parameters
        ----------
        new_data : pandas DataFrame
            The new data to be labeled.
            
        Returns
        -------
        y_prediction : list
            list of predicted labels.
        prob : list
            The probability that the label is correct.
        """
        new_data_X = self.preprocess.transform(new_data_X)
        y_prediction = self.ml_model.predict(new_data_X)
        y_probabilities = self.ml_model.predict_proba(new_data_X)
        y_probabilities = [max(prob) for prob in y_probabilities]
        return y_prediction, y_probabilities
    
    def K_fold(self):
        """
        This function performs a 10-fold cross-validation and returns the accuracies of each fold. 
            
        Returns
        -------
        accuracies : list
            The 10 accuracy values using 10-fold cross-validation.
        """
        from sklearn.model_selection import cross_val_score
        accuracies = cross_val_score(self.ml_classifier, self.X, self.y, cv=10)    
        return accuracies

class Active_ML_Model:
    """
    This class creates an active learning model based on the data sent, 
    data preprocessing, and type of ml classifier.
    
    """
    def __init__(self, data, ml_classifier, preprocess, n_samples = 5):
        """
        This function controls the initial creation of the active learning model.
        
        Parameters
        ----------
        data : pandas DataFrame
            The data the active learning model will be built on.
        ml_classifier : classifier object
            The classifier to be used to create the machine learning model.
        preprocess : Python Function
            The function used to preprocess the data before model creation.
        n_samples : int
            The number of random samples to be used in the initial model creation.
            
        Attributes
        -------
        ml_classifier : classifier object
            The classifier to be used to create the active learning model.
        preprocess : Python Function
            The function used to preprocess the data before model creation.
        test : pandas DataFrame
            The training set.
        train : pandas DataFrame
            The train set.
        """
        from sklearn.utils import shuffle
        data = shuffle(data)                                                    
        self.sample = data.iloc[:n_samples, :]   
        self.test = data.iloc[n_samples:, :]                                   
        self.train = None
        self.ml_classifier = ml_classifier
        self.preprocess = preprocess
    
    def Train(self, sample):
        """
        This function trains the innitial ml_model

        Parameters
        ----------
        train : pandas DataFrame
            The training set with labels
        
        Attributes Added
        ----------------
        ml_model : fitted machine learning classifier
            The machine learning model created using the training data.
        """
        import pandas as pd
        if self.train != None:
            self.train = pd.concat([self.train, sample])
        else:
            self.train = sample
        self.ml_model = ML_Model(self.train, self.ml_classifier, self.preprocess)

    def Continue(self, sampling_method, n_samples = 5):
        """
        This function continues the active learning model to the next step.
        
        Parameters
        ----------
        sampling_method : Python Function
            Determines the next set of samples to send to user.
        n_samples : int
            The number of samplest that should be added the the train set.
            
        Attributes Updated
        -------
        ml_classifier : classifier object
            The classifier to be used to create the active learning model.
        test : pandas DataFrame
            The training set.
        train : pandas DataFrame
            The train set.
        """
        import pandas as pd
        self.sample, self.test = sampling_method(self.ml_model, n_samples)
    
    def infoForProgress(self):
        y_actual = self.ml_model.train['y_value']
        y_pic = list(self.ml_model.train.index)
        y_pred, y_prob = self.ml_model.GetKnownPredictions(self.ml_model.train)
        y_pred = list(y_pred)
        correct_pic = []
        incorrect_pic = []
        for y_idx, y in enumerate(y_actual):
            if y == y_pred[y_idx]:
                correct_pic.append(y_pic[y_idx])
            else:
                incorrect_pic.append(y_pic[y_idx])
        return correct_pic, incorrect_pic
    
    def infoForResults(self):
        correct_pic, incorrect_pic = self.infoForProgress()
        test_pic = list(self.ml_model.train.idx)
        y_pred, y_prob = self.ml_model.GetUnknownPredictions(self.ml_model.test)
        health_pic = []
        blight_pic = []
        for y_idx, y in enumerate(y_pred):
            if y == 'H':
                health_pic.append(test_pic[y_idx])
            elif y == 'B':
                blight_pic.append(test_pic[y_idx])                
        return correct_pic, incorrect_pic, health_pic, blight_pic