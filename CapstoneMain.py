# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:14:18 2020

@author: Donovan
"""

from ML_Class import ML_Model
from DataPreprocessing import DataPreprocessing
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

preprocess = DataPreprocessing(True)
ml_classifier = RandomForestClassifier()

file_name = 'csvOut.csv'
data = pd.read_csv(file_name, index_col = 0, header = None)

corn_model = ML_Model(data.iloc[:-2, :], ml_classifier, preprocess)

accuracies = corn_model.K_fold()
print('K_fold Accuracies: ' + str(accuracies))
average_accuracy = accuracies.mean()
print('K_fold Average Accuracy: ' + str(average_accuracy))
predict, prob = corn_model.GetUnknownPredictions(data.iloc[198:, :-1])
print('Test Prediction: (' + predict[0] + ', ' + str(prob[0]) + ') (' + predict[1] + ', ' + str(prob[1]) + ')')