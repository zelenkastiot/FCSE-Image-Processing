import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('flowers.jpg')

imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

mask = np.zeros(img.shape[:2], dtype=np.uint8)
mask[130:620, 180:730] = 255

print(img.shape)

end = cv2.bitwise_and(imggray, imggray, mask=mask)

hist = cv2.calcHist([imggray], [0], mask, [256], [0, 256])
hist2 = cv2.calcHist([end], [0], mask, [256], [0, 256])

plt.figure(1)
plt.subplot(3, 2, 1)
slika = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(slika)

plt.subplot(3, 2, 2)
plt.imshow(imggray, cmap='gray')

plt.subplot(3, 2, 3)
plt.imshow(end, cmap='gray')

plt.subplot(3, 2, 4)
plt.plot(hist)

plt.subplot(3, 2, 5)
sobelx64f_x = cv2.Sobel(imggray, cv2.CV_64F, 1, 0, ksize=5)
abs_sobel64f_x = np.absolute(sobelx64f_x)
abs_sobel64f_x = abs_sobel64f_x / np.amax(abs_sobel64f_x[:])

plt.imshow(abs_sobel64f_x, cmap='gray')

plt.subplot(3, 2, 6)
sobelx64f_y = cv2.Sobel(imggray, cv2.CV_64F, 0, 1, ksize=5)
abs_sobel64f_y = np.absolute(sobelx64f_y)
abs_sobel64f_y = abs_sobel64f_y / np.amax(abs_sobel64f_y[:])

plt.imshow(abs_sobel64f_y, cmap='gray')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
