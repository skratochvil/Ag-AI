#A file to hold various feature selection methods.

import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


def extraTreesClassifier():
    with open('csvOut.csv', 'r') as f:
        data = pd.read_csv('csvOut.csv')

    X = data.iloc[:,1:27]  #feature columns
    y = data.iloc[:,-1]    #target column (last column)

    model = ExtraTreesClassifier()
    model.fit(X,y)
    print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()
    
extraTreesClassifier()