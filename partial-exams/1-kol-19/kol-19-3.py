"""

 Created on 15-Apr-21
 @author: Kiril Zelenkovski

"""
import cv2
import numpy as np

img = cv2.imread("Figure3.jpg", 1)

# BGR: red is 0, 0, 255
red = np.uint8([[[0, 0, 255]]])
hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
print(hsv_red)

# define range of blue color in HSV - rgb
lower_blue = np.array([-10, 100, 100])
upper_blue = np.array([10, 255, 255])

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
res = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Result", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
