import cv2
from matplotlib import pyplot as plt

img = cv2.imread('messi5.jpg', 1)

plt.subplot(131),plt.imshow(img[:,:,0], 'gray')
plt.subplot(132),plt.imshow(img[:,:,1], 'gray')
plt.subplot(133),plt.imshow(img[:,:,2], 'gray')
plt.show()
