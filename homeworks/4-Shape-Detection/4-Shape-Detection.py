"""

 Created on 28-Apr-21
 @author: Kiril Zelenkovski
 @topic: Homework Assignment 4

    Using the function cv2.matchShapes() we do a search on all the images that are in the database.zip folder.
    As a input to the search we pass on an image from the query.zip folder and the result is a list (sorted) of all the
    images ranked by similarity with the query image.
"""
import sys
from os import listdir, walk
from os.path import isfile, join
import cv2
import numpy as np
import operator
import os


# Useful function 1
def get_imageName(file):
    """
    Function for getting file name from path.

    :param file: file path of image
    :return: only the name of file (without .png, .jpg etc.)
    """
    s = os.getcwd() + "\\database\\" + file
    base = os.path.basename(s)
    return os.path.splitext(base)[0]

# Useful function 2
def get_all_contours(img):
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)
    return contours

# Useful function 3
def get_ref_contour(img):
    ref_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(ref_gray, 127, 255, 0)

    # Find all the contours in the threshold image. The values for the second and third parameters are restricted
    # to a # certain number of possible values. You can learn more # 'findContours' function here:
    # http://docs.opencv.org/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
    contours, hierarchy = cv2.findContours(thresh, 1, 2)

    # Extract the relevant contour based on area ratio.
    # We use the # area ratio because the main image boundary contour is # extracted as well and we don't want that.
    # This area ratio # threshold will ensure that we only take the contour inside # the image.
    for contour in contours:
        area = cv2.contourArea(contour)
        img_area = img.shape[0] * img.shape[1]
        if 0.05 < area / float(img_area) < 0.8:
            return contour


# Main function 1: Implemented query search for similarity
def query_image_search(query):
    """

    :param query: input reference image from the query.zip directory
    :return: sorted list of tuple (dict paris) containing all the images from the database that ranked by similarity

    The idea is too keep a dictionary (key, value) for similarity where:
        - value: number that is returned from the cv2.matchShapes() function = similarity with query image.
                 the smaller the number the more similar two images are
        - key: picture from th database directory to associate with the similarity

    """
    # Calculation of the contours on input image
    ret, thresh = cv2.threshold(query, 100, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt1 = contours[len(contours) - 1]

    # Define the path from where we read the images for the resulting list
    my_path = "database"
    images = [f for f in listdir(my_path) if isfile(join(my_path, f))]

    # Initialize all the values in the dictionary to 0
    images_dict = {}
    for img in images:
        images_dict[img] = 0

    # Iteration through all the images
    for img in images:
        img2 = cv2.imread(my_path + "\\" + img, 0)

        # For each read image calculate the contour
        ret, thresh2 = cv2.threshold(img2, 100, 255, 0)
        contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnt2 = contours[len(contours) - 1]

        # Compare the contours, store the similarity result as value in the dictionary
        ret = cv2.matchShapes(cnt1, cnt2, 1, 0.0)
        images_dict[img] = ret

    # Sort the dictionary (most similar to least with the query image)
    sorted_images = sorted(images_dict.items(), key=operator.itemgetter(1))
    print(sorted_images)



if __name__ == '__main__':
    # Reading and saving the image names from the query reference directory
    files = []
    for (dir_path, dir_names, filenames) in walk("query"):
        files.extend(filenames)
        break

    # Iterate through all the images and display the list of similarity
    for pic in files:
        query_img = cv2.imread('query/' + pic, 0)
        plotName = get_imageName(pic)
        print("-_-_-_-_-_-_-_-_-_-_-_-_ Similarity  list for image: " + plotName + ".jpg" + " -_-_-_-_-_-_-_-_-_-_-_-_")
        query_image_search(query_img)
        print()
        print("-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-")
        print()
