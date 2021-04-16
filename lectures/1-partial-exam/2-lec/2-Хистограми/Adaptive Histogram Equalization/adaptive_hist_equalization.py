import cv2
import numpy as np
from matplotlib import pyplot as plt

#Adaptive histogram equalization

# Load test image
img = cv2.imread('parrot.jpg', 0)
#img = cv2.imread('dental.jpg', 0)
#img = cv2.imread('skull.jpg', 0)
#img = cv2.imread('moon.jpg', 0)


#Apply global and adaptive histogram equalization
eqImg = cv2.equalizeHist(img)


# create a CLAHE object (Arguments are optional).
#Contrast Limited Adaptive Histogram Equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
claheImg1 = clahe.apply(img)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
claheImg2 = clahe.apply(img)

# In a small area, histogram would confine to a small region (unless there is noise).
# If noise is there, it will be amplified. To avoid this, contrast limiting is applied.
# If any histogram bin is above the specified contrast limit (by default 40 in OpenCV),
# those pixels are clipped and distributed uniformly to other bins before applying histogram equalization.
# After equalization, to remove artifacts in tile borders, bilinear interpolation is applied.

# Show images
plt.figure(1)

plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('Original image')

plt.subplot(222)
plt.imshow(eqImg, cmap='gray')
plt.title('Global histogram')

plt.subplot(223)
plt.imshow(claheImg1, cmap='gray')
plt.title('Tiling 8*8 histograms')

plt.subplot(224)
plt.imshow(claheImg2, cmap='gray')
plt.title('Tiling 16*16 histograms')

plt.show()

# Save images
cv2.imwrite('CLAHE_global.png', eqImg)
cv2.imwrite('CLAHE_8by8.png', claheImg1)
cv2.imwrite('CLAHE_16by16.png', claheImg2)
