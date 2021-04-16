import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('berndsface.png', 0)

sobelx64f_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
abs_sobel64f_x = np.absolute(sobelx64f_x)
abs_sobel64f_x = abs_sobel64f_x / np.amax(abs_sobel64f_x[:])

sobelx64f_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
abs_sobel64f_y = np.absolute(sobelx64f_y)
abs_sobel64f_y = abs_sobel64f_y / np.amax(abs_sobel64f_y[:])

edgeSum = np.sqrt(np.power(abs_sobel64f_x, 2.0) + np.power(abs_sobel64f_y, 2.0))
ret, thresh = cv2.threshold(edgeSum, 0.15, 1, cv2.THRESH_BINARY)

plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(thresh, cmap='gray')
plt.title('Sobel edge'), plt.xticks([]), plt.yticks([])

plt.show()
