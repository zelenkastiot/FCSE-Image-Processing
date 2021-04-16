import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load test image
img = cv2.imread('croppedBike.png');
# Construct filter impulse responses
h1 = np.ones((21, 21), np.float32) / 441
h2 = np.ones((1, 5), np.float32) / 5
h3 = np.ones((5, 1), np.float32) / 5
h4 = np.array(([0, 0, 0], [0, 2, 0], [0, 0, 0]), dtype="int")
h5 = np.ones((3, 3), np.float32) / 9
# Perform filtering
filteredImg1 = cv2.filter2D(img, -1, h1)
filteredImg2 = cv2.filter2D(img, -1, h2)
filteredImg3 = cv2.filter2D(img, -1, h3)
filteredImg4 = cv2.filter2D(img, -1, h4 - h5)
# Show images
plt.figure(1)

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(filteredImg1)

plt.figure(2)

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(filteredImg2)

plt.figure(3)

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(filteredImg3)

plt.figure(4)

plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(filteredImg4)

plt.show()
# Save images
cv2.imwrite('Convolution_1.png', filteredImg1)
cv2.imwrite('Convolution_2.png', filteredImg2)
cv2.imwrite('Convolution_3.png', filteredImg3)
cv2.imwrite('Convolution_4.png', filteredImg4)
