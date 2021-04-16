import cv2
from matplotlib import pyplot as plt

# img = cv2.imread('test1.jpg')
img = cv2.imread('image2.jpg', 0)
# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
histr = cv2.calcHist([img], [0], None, [256], [0, 256])
print(histr)
plt.plot(histr, color='gray')
plt.xlim([0, 256])
plt.show()
