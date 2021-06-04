"""

 Created on 24-Apr-21
 @author: Kiril Zelenkovski
 @topic: Homework Assignment 3

For all the images in database.zip there is an implementation of algorithms for thresholding the image where the
leaves are clearly displayed. For better results on some of the images a Gauss filter was applied in order to improve
the Otsu's thresholding algorithm.

After the images were segmented on each one there is an implementation of edge detection.

All the results, for the segmentation and the contour detection are in folders threshold analysis and contour analysis
respectively.

"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
from os import walk
import os

# Matplotlib design features
plt.rcParams['font.size'] = 7
plt.rcParams['savefig.edgecolor'] = "0.2"
plt.rcParams['savefig.facecolor'] = "0.95"

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
def get_contours(image):
    """
    Function for contour extraction of color image.

    :param image: bgr color image
    :return: contours of Otsu threshold image
    """
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the input image with Otsu
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    ret3, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Test with adaptive thresholding
    # th3 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # th4 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Find the contours in the above image
    contours, hierarchy = cv2.findContours(th, 2, 1)

    return contours

# Main analysis 1: image segmentation
def plot_thresholds(image, savingName):
    """
    Thresholding and image segmentation analysis.

    :param image: grayscale image
    :param savingName: string, name of image. Used when generating new plots


    Thresholding Analysis explained:

       1 - Apply a basic binary threshold (def value 127), Otsu's on noisy and filtered
       2-Темплејти-пирамиди-на-слики Compare the threshold values before and after filter is applied. In almost all
       the cases (27 out of 30 images) the result of the Otsu's Thresholding on a noisy
       image and filtered gave same results. The 3 images had a very small difference in
       the thresholds.

       All the plots are saved inside the "threshold-analysis" directory.

    """
    # Global thresholding
    ret1, th1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Otsu's thresholding
    ret2, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    th4 = cv2.cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    # th4 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Print thresholds to see difference
    print(savingName + ".jpg: Thresholding analysis")
    print("Default threshold value: 127")
    print("Otsu's Thresholding (noisy image) : ", ret2)
    print("Otsu's Thresholding (after Gaussian filter) : ", ret3)
    print()

    # Plot all the images and their histograms
    images = [image, 0, th1,
              image, 0, th2,
              blur, 0, th3,
              image, 0, th4]

    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding",
              'Original Noisy Image', "Histogram", 'Adaptive Thresholding']

    for i in range(4):
        plt.subplot(4, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(4, 3, i * 3 + 2), plt.hist(images[i * 3].flatten(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(4, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])

    # Save results in directory
    plt.suptitle(savingName + ".jpg")
    plt.tight_layout()
    plt.savefig("threshold-analysis/" + savingName + "-thresholds.png")
    plt.show()

# Main analysis 2: contours detection
def plot_contours(file, savingName):
    """
    Function for contour detection.

    :param file: path to file
    :param savingName: string, name of image. Used when generating new plots

    All the plots are saved inside the "contour-analysis" directory.
    """
    # Read image
    img_bgr = cv2.imread('database/' + file)
    img_gry = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img_gry, (5, 5), 0)
    ret3, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Iterate over the extracted contours
    for contour in get_contours(img_bgr):
        # Extract convex hull from the contour
        hull = cv2.convexHull(contour, returnPoints=False)
        hull[::-1].sort(axis=0)

        # Extract convexity defects from the above hull
        defects = cv2.convexityDefects(contour, hull)

        if defects is None:
            continue

        # Draw lines and circles to show the defects
        for i in range(defects.shape[0]):
            start_defect, end_defect, far_defect, _ = defects[i, 0]
            far = tuple(contour[far_defect][0])
            cv2.circle(img_bgr, far, 5, [128, 0, 0], -1)
            cv2.drawContours(img_bgr, [contour], -1, (0, 0, 0), 3)

    # Plot figure
    plt.figure(1)

    plt.subplot(1, 2, 1)
    plt.title("Segmented image")
    plt.imshow(img_bgr)

    plt.subplot(1, 2, 2)
    plt.title("Otsu's threshold")
    plt.imshow(th, cmap='gray')

    plt.tight_layout()
    plt.suptitle(savingName + ".jpg")
    plt.savefig("contour-analysis/" + savingName + "-contours.png")
    plt.show()


if __name__ == '__main__':
    # Reading and saving the image names from the database
    files = []
    for (dir_path, dir_names, filenames) in walk("database"):
        files.extend(filenames)
        break

    print("-=-=-=-=-=-=-=-=-=-= START: Thresholding analysis -=-=-=-=-=-=-=-=-=-= \n")
    for pic in files:
        img = cv2.imread('database/' + pic, 0)
        plotName = get_imageName(pic)
        plot_thresholds(img, plotName)

    print("-=-=-=-=-=-=-=-=-=-= END: Thresholding analysis =-=-=-=-=-=-=-=-=-=-=- \n \n")

    print("-=-=-=-=-=-=-=-=-=-= START: Contour detection-=-=-=-=-=-=-=-=-=-= \n")
    for pic in files:
        plotName = get_imageName(pic)
        plot_contours(pic, plotName)

    print("-=-=-=-=-=-=-=-=-=-= END: Contour detection =-=-=-=-=-=-=-=-=-=-=")
