import os
import csv
import numpy
from skimage.io import imread, imshow

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
        
def getBasicFeatures(listOfImagePaths, csvOutFileName):
    lowRed = 180
    highRed = 250
    lowGreen = 160
    highGreen = 200
    lowBlue = 150
    highBlue = 240
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
        redMean = numpy.mean(red)     

        green = image[0:4000, 0:6000, 1]
        green = numpy.reshape(green, (6000 * 4000))
        greenMean = numpy.mean(green)

        blue = image [0:4000, 0:6000, 2]
        blue = numpy.reshape(blue, (6000 * 4000))
        blueMean = numpy.mean(blue)
        
# =============================================================================
# Calculate features: number of pixels in each band with values within a certain range.
# This color range represents the "brown" of the blights and was determined manually
# by inspecting individual "blighted" pixels on many of our images using this website: https://pinetools.com/image-color-picker
# Please note that these threshold values can probably be tuned more.
# Nice-To-Have: find a way to count pixels for which all three RGB values are within the threshold, which would be better
# than simply counting the values independently of each other. 
# This was tried by iterating over each pixel, but it totally destroyed processing time.
# =============================================================================
        numBrownRed,bin_edges = numpy.histogram(red, bins=1, range=(lowRed,highRed))
        numBrownGreen, bin_edges = numpy.histogram(green, bins=1, range=(lowGreen,highGreen))
        numBrownBlue, bin_edges = numpy.histogram(blue, bins=1, range=(lowBlue,highBlue))
#Create feature vectors as list of tuples
        featureVector = (imagePath, gray, redMean, greenMean, blueMean, numBrownRed[0], numBrownGreen[0], numBrownBlue[0])
        csvOut.append(featureVector)
#Save feature vectors as CSV file        
    csvfile = open(csvOutFileName,'w', newline='')
    obj = csv.writer(csvfile)
    obj.writerow(('file', 'AvgGrayValue', 'AvgRedValue', 'AvgGreenValue', 'AvgBlueValue', 'NumRedBrown', 'NumGreenBrown', 'NumBlueBrown'))
    obj.writerows(csvOut)
    print("Feature extraction complete.")

#Example usage. 
#testImages = getHealthyImagePaths()    
#getBasicFeatures(testImages[0:5], 'csvOut.csv')








