import cv2
import numpy as np
from matplotlib import pyplot as plt

# Histogram equalization example
# Load test image
#img = cv2.cvtColor(cv2.imread('bay.jpg'),cv2.COLOR_BGR2GRAY)
#img = cv2.cvtColor(cv2.imread('brain.jpg'),cv2.COLOR_BGR2GRAY)
#img = cv2.cvtColor(cv2.imread('Example.png'), cv2.COLOR_BGR2GRAY)
img = cv2.cvtColor(cv2.imread('moon.jpg'), cv2.COLOR_BGR2GRAY)

# Perform histogram equalization
eqImg = cv2.equalizeHist(img)

# Show images
cv2.namedWindow("Original image", cv2.WINDOW_AUTOSIZE)
cv2.imshow('Original image', img)
cv2.waitKey(0)
cv2.namedWindow("After histogram equalization", cv2.WINDOW_AUTOSIZE)
cv2.imshow('After histogram equalization', eqImg)
cv2.waitKey(0)
cv2.imwrite('Histogram_Equalization_eqImg.png', eqImg)

plt.figure(1)
plt.subplot(211)
# Show histogram for original image
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('histogram'), loc='upper left')

plt.subplot(212)
# Show histogram for histogram-equalized image
hist, bins = np.histogram(eqImg.flatten(), 256, [0, 256])
plt.hist(eqImg.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('histogram'), loc='upper left')
plt.show()

cv2.destroyAllWindows()
