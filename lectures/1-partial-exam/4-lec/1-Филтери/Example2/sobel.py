import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load test image
img = cv2.imread('croppedBike.png')
# Construct filter impulse responses
h1 = np.array(([1, 0, -1], [2, 0, -2], [1, 0, -1]), dtype="int")
# Perform filtering
filteredImg1 = cv2.filter2D(img, -1, h1)
# Show images
plt.figure(1)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(filteredImg1)
plt.show()
# Save images
cv2.imwrite('Convolution_1.png', filteredImg1)
