import cv2
import numpy as np
from matplotlib import pyplot as plt

# Example histogram

# Load test image
img = cv2.imread('bay.jpg')

plt.figure(1)
plt.subplot(1, 2, 1)
# flatten() - Return a copy of the array collapsed into one dimension
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
plt.hist(img.flatten(), 256, [0, 256], color='gray')
plt.xlim([0, 256])
plt.legend('histogram', loc='upper left')
plt.title('Image histogram')
plt.subplot(1, 2, 2)
plt.title('Original image')
plt.imshow(img)
plt.show()
