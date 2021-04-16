import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('letter.jpg')
rows, cols, ch = img.shape

pts1 = np.float32([[157, 139], [542, 139], [95, 686], [620, 686]])
pts2 = np.float32([[0, 0], [1000, 0], [0, 1000], [1000, 1000]])

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(img, M, (1000, 1000))

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()
