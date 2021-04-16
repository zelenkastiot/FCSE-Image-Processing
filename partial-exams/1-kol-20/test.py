"""

 Created on 15-Apr-21
 @author: Kiril Zelenkovski

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("flowers.jpg", 1)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create mask
mask = np.zeros(img_gry.shape[:2], np.uint8)
mask[130:620, 180:720] = 255
masked_img = cv2.bitwise_and(img_gry, img_gry, mask=mask)

# Calculate histogram with mask
hist_mask = cv2.calcHist([img_gry], [0], mask, [256], [0, 256])

# Calculate Sobel operator for x (1, 0)
sobelx64fx = cv2.Sobel(img_gry, cv2.CV_64F, 1, 0, ksize=5)
abs_sobel64fx = np.absolute(sobelx64fx)
sobelx = abs_sobel64fx / np.amax(abs_sobel64fx[:])

# Calculate Sobel operator for y (0, 1)
sobelx64fy = cv2.Sobel(img_gry, cv2.CV_64F, 0, 1, ksize=5)
abs_sobel64fy = np.absolute(sobelx64fy)
sobely = abs_sobel64fy / np.amax(abs_sobel64fy[:])


plt.figure(1)
plt.subplot(3, 2, 1)
plt.imshow(img_rgb)

plt.subplot(3, 2, 2)
plt.imshow(img_gry, cmap='gray')

plt.subplot(3, 2, 3)
plt.imshow(masked_img, cmap='gray')

plt.subplot(3, 2, 4)
plt.plot(hist_mask)

plt.subplot(3, 2, 5)
plt.imshow(sobelx, cmap='gray')

plt.subplot(3, 2, 6)
plt.imshow(sobely, cmap='gray')

plt.show()
