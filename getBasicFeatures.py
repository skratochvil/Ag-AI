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
blightedImagePaths = []
for root, dirs, files in os.walk(os.path.abspath("images/blighted")):
    for file in files:
        blightedImagePaths.append(os.path.join(root, file))

#get file paths for healthy images        
healthyImagePaths = []
for root, dirs, files in os.walk(os.path.abspath("images/healthy")):
    for file in files:
        healthyImagePaths.append(os.path.join(root, file))
        
def getBasicFeatures(listOfImagePaths, csvOutFileName):
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
        featureVector = (imagePath, gray, red, green, blue)
        csvOut.append(featureVector)
#Save feature vectors as CSV file        
    csvfile = open(csvOutFileName,'w', newline='')
    obj = csv.writer(csvfile)
    obj.writerow(('file', 'gray', 'red', 'green', 'blue'))
    obj.writerows(csvOut)

#Example usage. 
#getBasicFeatures(healthyImagePaths[0:5], 'csvOut.csv')







