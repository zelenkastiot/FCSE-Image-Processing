"""

 Created on 15-Apr-21
 @author: Kiril Zelenkovski

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('Figure4.jpg', 0)

img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)

kernelx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)

img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

cv2.imshow("Prewitt X", img_prewittx)
cv2.imshow("Prewitt Y", img_prewitty)
cv2.waitKey(0)
cv2.destroyAllWindows()

