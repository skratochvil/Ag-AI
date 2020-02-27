# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:14:18 2020

@author: Donovan
"""

from ML_Class import ML_Model
from ImagePreprocessing import ImageProcessing
from DataPreprocessing import StandardScaling
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#folder_name = 'images/'
#ImageProcessing(folder_name)

file_name = 'csvOut.csv'

data = pd.read_csv(file_name, index_col = 0, header = None)
ml_classifier = RandomForestClassifier()
corn_model = ML_Model(data, ml_classifier, DataPreprocessing = StandardScaling)
accuracies = corn_model.K_fold()
average_accuracy = accuracies.mean()
print('K_fold Accuracy: ' + str(average_accuracy))
predict, prob = corn_model.GetUnknownPredictions(data.iloc[:2, :-1])
print(predict)