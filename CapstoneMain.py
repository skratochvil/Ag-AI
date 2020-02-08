# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:14:18 2020

@author: Donovan
"""

from ML_Class import ML_Model
from ImagePreprocessing import ImageProcessing
from DataPreprocessing import StandardScaling
from AssessmentTechnique import K_fold
from sklearn.svm import SVC
import pandas as pd

folder_name = 'image_folder/'
ImageProcessing(folder_name)

file_name = 'data_source.csv'

data = pd.read_csv(file_name)
ml_classifier = SVC(kernel = 'linear', random_state = 0)
Fruit_Model = ML_Model(data, ml_classifier, DataPreprocessing = StandardScaling)
accuracies = K_fold(Fruit_Model.ml_model)
average_accuracy = accuracies.mean()
print('K_fold Accuracy: ' + Fruit_Model.score)