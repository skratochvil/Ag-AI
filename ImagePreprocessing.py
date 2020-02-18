# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 11:20:09 2020

@author: Donovan
"""

import cv2
#conda install -c conda-forge opencv=3.4.1
#3-Clause BSD License

import os
import csv
import numpy
from skimage.io import imread, imshow
#conda install -c anaconda scikit-image
#BSD 3-Clause

def avgGray(image):
    grayscaleArray = numpy.reshape(image, -1)
    gray_mean = numpy.mean(grayscaleArray)
    return gray_mean

def avgRed(image):
    red = image[0:4000, 0:6000, 0]
    red = numpy.reshape(red, -1)
    red_mean = numpy.mean(red)
    return red_mean

def avgGreen(image):
    green = image[0:4000, 0:6000, 1]
    green = numpy.reshape(green, -1)
    green_mean = numpy.mean(green)
    return green_mean

def avgBlue(image):
    blue = image [0:4000, 0:6000, 2]
    blue = numpy.reshape(blue, -1)
    blue_mean = numpy.mean(blue)
    return blue_mean
    
def numBrownRed(image):
    red = image[0:4000, 0:6000, 0]
    red = numpy.reshape(red, -1)
    num_brown_red, bin_edges = numpy.histogram(red, bins=1, range=(180, 250))
    return num_brown_red[0]

def numBrownGreen(image):
    green = image[0:4000, 0:6000, 1]
    green = numpy.reshape(green, -1)
    num_brown_green, bin_edges = numpy.histogram(green, bins=1, range=(160, 200))
    return num_brown_green[0]

def numBrownBlue(image):
    blue = image [0:4000, 0:6000, 2]
    blue = numpy.reshape(blue, -1)
    num_brown_blue, bin_edges = numpy.histogram(blue, bins=1, range=(150, 240))
    return num_brown_blue[0]

def FdHuMoments(image):
    """
    Extracts Hu moments feature from an image
    Parameters
    ----------
    
    image : imread
        The image used for feature extraction
    Returns
    -------
    Feature : Float Array
        The Hu moments in the image.
    Reference
    ---------
    https://gogul.dev/software/image-classification-python
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def FdHaralick(image):
    import mahotas
    #
    #MIT License
    """
    Extracts Haralick texture feature from an image
    Parameters
    ----------
    
    image : imread
        The image used for feature extraction
    Returns
    -------
    Feature : Float Array
        The Haralick texture in the image.
    Reference
    ---------
    https://gogul.dev/software/image-classification-python
    """
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

def FdHistogram(image, mask=None, bins = 8):
    """
    Extracts color histogram feature from an image
    Parameters
    ----------
    
    image : imread
        The image used for feature extraction
    Returns
    -------
    Feature : Float Array
        The color histogram in the image.
    Reference
    ---------
    https://gogul.dev/software/image-classification-python
    """
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

import numpy as np
def ImageProcessing(folder_name):
    def allFilesInDir(dir_name, label):
        csvOut = []
        for root, dirs, files in os.walk(os.path.abspath(dir_name)):
            for file in files:

                image = imread(os.path.join(root, file), as_gray=True)
                gray_mean = avgGray(image)

                image = imread(os.path.join(root, file))
                red_mean = avgRed(image)
                green_mean = avgGreen(image)
                blue_mean = avgBlue(image)
                num_brown_red = numBrownRed(image)
                num_brown_green = numBrownGreen(image)
                num_brown_blue = numBrownBlue(image)
                
                image = cv2.imread(os.path.join(root, file))
                fv_hu_moments = FdHuMoments(image)
                fv_haralick = FdHaralick(image)
#                fv_histrogram = FdHistogram(image)

                feature_vector = np.hstack([file, fv_hu_moments, fv_haralick, gray_mean, red_mean, green_mean, blue_mean, num_brown_red, num_brown_green, num_brown_blue, label])
                
                csvOut.append(feature_vector)
        return csvOut
    
    blighted_features = allFilesInDir('images/blighted', 'B')
    healthy_features = allFilesInDir('images/healthy', 'H')
    csvfile = open('csvOut.csv','w', newline = '')
    obj = csv.writer(csvfile)
    obj.writerows(blighted_features)
    obj.writerows(healthy_features)