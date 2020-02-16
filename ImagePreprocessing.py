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

# =============================================================================
# Retrieves basic image features as .csv file
# Input: list of file paths for input images, CSV file name for output
# Output: average pixel intensity for gray, red, blue, and green as CSV file
# Uncomment bottom line of file to see sample usage.
# =============================================================================

#get file paths for blighted images
def getBlightedImagePaths():
    blightedImagePaths = []
    for root, dirs, files in os.walk(os.path.abspath("images/blighted")):
        for file in files:
            blightedImagePaths.append(os.path.join(root, file))
    return blightedImagePaths

#get file paths for healthy images        
def getHealthyImagePaths():
    healthyImagePaths = []
    for root, dirs, files in os.walk(os.path.abspath("images/healthy")):
        for file in files:
            healthyImagePaths.append(os.path.join(root, file))
    return healthyImagePaths
        
def getBasicFeatures(listOfImagePaths, csvOutFileName, label):
    csvOut = []
    for imagePath in listOfImagePaths:
# =============================================================================
# Calculate feature: average intensity of grayscale pixels
# =============================================================================
#load image as array of pixels in grayscale
        image = imread(imagePath, as_gray=True)
    #uncomment line below to view image
    #image.shape, imshow(image)
#convert to 2d array of gray pixel values
        grayscaleArray = numpy.reshape(image, (6000 * 4000))
        gray = numpy.mean(grayscaleArray)
# =============================================================================
# Calculate features: average intensity of red, blue, and green pixels
# =============================================================================
#loads image as 3D array [X,Y,Z] where Z represents one of the color bands
        image = imread(imagePath)
#Splits RGB array into color bands and calculates average pixel intensity for each band
        red = image[0:4000, 0:6000, 0]
        red = numpy.reshape(red, (6000 * 4000))
        red = numpy.mean(red)

        green = image[0:4000, 0:6000, 1]
        green = numpy.reshape(green, (6000 * 4000))
        green = numpy.mean(green)

        blue = image [0:4000, 0:6000, 2]
        blue = numpy.reshape(blue, (6000 * 4000))
        blue = numpy.mean(blue)
#Create feature vectors as list of tuples
        featureVector = (imagePath, gray, red, green, blue, label)
        csvOut.append(featureVector)
#Save feature vectors as CSV file        
    csvfile = open(csvOutFileName,'a', newline='')
    obj = csv.writer(csvfile)
    obj.writerow(('file', 'gray', 'red', 'green', 'blue', 'label'))
    obj.writerows(csvOut)

#Example usage. 
#testImages = getHealthyImagePaths()    
#getBasicFeatures(testImages[0:5], 'csvOut.csv')
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


                image = cv2.imread(os.path.join(root, file))
                fv_hu_moments = FdHuMoments(image)
                fv_haralick = FdHaralick(image)
                fv_histrogram = FdHistogram(image)

                feature_vector = np.hstack([file, fv_histrogram, fv_haralick, fv_hu_moments, label])
                
                csvOut.append(feature_vector)
        return csvOut
    
    blighted_features = allFilesInDir('images/blighted', 'B')
    healthy_features = allFilesInDir('images/healthy', 'H')
    csvfile = open('csvOut.csv','w', newline = '')
    obj = csv.writer(csvfile)
    obj.writerows(blighted_features)
    obj.writerows(healthy_features)