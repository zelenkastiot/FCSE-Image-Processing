"""

 Created on 17-Apr-21
 @author: Kiril Zelenkovski [161141]
 @topic: 1st partial exam

"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("flowers.jpg", 1)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# create a mask
mask = np.zeros(img_gry.shape[:2], np.uint8)
mask[130:620, 180:730] = 255
masked_img = cv2.bitwise_and(img_gry, img_gry, mask=mask)

# Calculate histogram with mask and without mask
# Check third argument for mask
hist_mask = cv2.calcHist([img_gry], [0], mask, [256], [0, 256])

# Calculate sobel x: 1, 0
sobelx64f_x = cv2.Sobel(img_gry, cv2.CV_64F, 1, 0, ksize=5)
abs_sobel64f_x = np.absolute(sobelx64f_x)
abs_sobel64f_x = abs_sobel64f_x / np.amax(abs_sobel64f_x[:])

# Calculate sobel y: 0, 1
sobelx64f_y = cv2.Sobel(img_gry, cv2.CV_64F, 1, 0, ksize=5)
abs_sobel64f_y = np.absolute(sobelx64f_y)
abs_sobel64f_y = abs_sobel64f_y / np.amax(abs_sobel64f_y[:])

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
plt.imshow(abs_sobel64f_x, cmap='gray')

plt.subplot(3, 2, 6)
plt.imshow(abs_sobel64f_y, cmap='gray')

plt.show()
