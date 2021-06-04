import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np


# Input is a color image
def get_contours(img):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the input image
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)

    # Find the contours in the above image
    contours, hierarchy = cv2.findContours(thresh, 2, 1)

    return contours


if __name__ == '__main__':
    img = cv2.imread("1.jpg")
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img_gry, (5, 5), 0)
    ret3, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Iterate over the extracted contours
    for contour in get_contours(img):
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
            start = tuple(contour[start_defect][0])
            end = tuple(contour[end_defect][0])
            far = tuple(contour[far_defect][0])
            cv2.circle(img, far, 5, [128, 0, 0], -1)
            cv2.drawContours(img, [contour], -1, (0, 0, 0), 3)

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(img)

    plt.subplot(1, 2, 2)
    plt.imshow(th, cmap='gray')
    plt.show()

    # cv2.imshow('Convexity defects', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
