# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 13:30:29 2020

This file is used to test the Active Learning model.

@author: Donovan
"""

from ML_Class import Active_ML_Model
from DataPreprocessing import DataPreprocessing
from sklearn.ensemble import RandomForestClassifier
from SamplingMethods import lowestPercentage
import pandas as pd

preprocess = DataPreprocessing(True)
ml_classifier = RandomForestClassifier()
sampling_method = lowestPercentage

file_name = 'csvOut.csv'
data = pd.read_csv(file_name, index_col = 0, header = None)
data = data.iloc[:, :-1]
corn_active_model = Active_ML_Model(data, ml_classifier, preprocess)

stop = False
while stop == False:
    corn_active_model.Continue(sampling_method)
    stop = True