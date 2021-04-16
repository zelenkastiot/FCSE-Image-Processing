import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('berndsface.png')
img = cv2.GaussianBlur(img, (5, 5), 0)
img = np.float32(img)

# Apply Prewitt filter in vertical direction
h = np.array(([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]), dtype="float32")
filteredImg1 = cv2.filter2D(img, -1, h)
filteredImg1 = abs(filteredImg1)
filteredImg1 = filteredImg1 / np.amax(filteredImg1[:])

# Apply Prewitt filter in horizontal direction
h = np.array(([-1, -1, -1], [0, 0, 0], [1, 1, 1]), dtype="float32")
filteredImg2 = cv2.filter2D(img, -1, h)
filteredImg2 = abs(filteredImg2)
filteredImg2 = filteredImg2 / np.amax(filteredImg2[:])

edgeSum = np.sqrt(np.power(filteredImg1, 2.0) + np.power(filteredImg2, 2.0))
ret, thresh = cv2.threshold(edgeSum, 0.15, 1, cv2.THRESH_BINARY)

# Show images
plt.subplot(131)
plt.imshow(filteredImg1)
plt.title('Prewitt Vertical')

plt.subplot(132)
plt.imshow(filteredImg2)
plt.title('Prewitt Horizontal')

plt.subplot(133)
plt.imshow(thresh)
plt.title('Prewitt')
plt.show()

