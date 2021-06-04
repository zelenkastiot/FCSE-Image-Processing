"""

 Created on 03-Jun-21
 @author: Kiril Zelenkovski [161141]
 @topic: 2nd partial exam

"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

if __name__ == '__main__':
    img = cv2.imread('car.jpg', 0)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    img_threshold = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU)[1]

    # 1: Edge detection using Dilatation
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_dilated = cv2.dilate(img_blur, se, iterations=1)

    # Subtraction / thresholding
    height, width = img_blur.shape
    edge = np.zeros((height, width), np.uint8)
    cv2.absdiff(img_dilated, img_blur, edge)
    _, difference = cv2.threshold(edge, 0, 255, cv2.THRESH_OTSU)

    plt.figure(1)
    plt.suptitle("1: Edge detection using Dilatation")

    plt.subplot(2, 2, 1)
    plt.title("Gaussian filter")
    plt.imshow(img_blur, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Dilatation")
    plt.imshow(img_dilated, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Dilatation - Original")
    plt.imshow(edge, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Edge detection")
    plt.imshow(difference, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 2: Morphological operations
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # Opening: Erosion then Dilatation
    opening = cv2.morphologyEx(img_threshold, cv2.MORPH_OPEN, se)
    # Closing: Dilatation then Erosion
    closing = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, se)

    plt.figure(1)
    plt.suptitle("2: Morphological operations")

    plt.subplot(1, 2, 1)
    plt.title("Opening")
    plt.imshow(opening, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Closing")
    plt.imshow(closing, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 3.1: Contour detection (convexity defects)
    img = cv2.imread("car.jpg")
    contours, hierarchy = cv2.findContours(img_threshold, 2, 1)

    for contour in contours:
        # Extract hull from current contour
        hull = cv2.convexHull(contour, returnPoints=False)
        hull[::-1].sort(axis=0)

        # Extract convexity defects from current hull
        defects = cv2.convexityDefects(contour, hull)

        if defects is None:
            continue

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            cv2.circle(img, far, 5, [128, 0, 0], -1)
            cv2.drawContours(img, [contour], -1, (0, 0, 0), 3)

    plt.figure(1)
    plt.title("3.1: Contour detection (convexity defects)")
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 3.2: Contour hierarchy
    img = cv2.imread('car.jpg')

    # Iterate and draw rects around contours
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 12, 12), 2)

    plt.figure(1)
    plt.title("3.2: Contour detection (hierarchy)")
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # 4: Display results (extract plate) using Algorithm
    img = cv2.imread("car.jpg", 0)
    height, width = img.shape
    edge = np.zeros((height, width), np.uint8)
    cv2.absdiff(img_dilated, img, edge)
    _, bw = cv2.threshold(edge, 20, 255, cv2.THRESH_BINARY)
    contours2, hierarchy2 = cv2.findContours(bw, 1, 2)

    solidity_values = []

    for c in contours2:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        area_contour = cv2.contourArea(approx)
        solidity_values.append(area_contour)

    solidity_values = np.array(solidity_values)
    my_contour = np.argmax(solidity_values)

    # Draw plate contour
    img = cv2.imread("car.jpg")
    cv2.drawContours(img, [contours2[my_contour]], -1, (255, 0, 0), 2)
    # Make plate mask
    x, y, w, h = cv2.boundingRect(contours2[my_contour])
    plate = img[y: y + h, x: x + w]

    plt.figure(1)
    plt.suptitle("4: Contour extraction")

    plt.subplot(1, 2, 1)
    plt.title("Original image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Extracted plate")
    plt.imshow(plate, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 5: Save car_plate.jpg
    plt.figure(1)
    plt.title("5: Extracted plate")
    plt.imshow(img)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig("car_plate.jpg")
    # plt.show()

    # 6: Save plate.jpg
    plt.figure(1)
    plt.title("6: Car plate")
    plt.imshow(plate)
    plt.tight_layout()
    plt.axis('off')
    plt.savefig("plate.jpg")
    # plt.show()
