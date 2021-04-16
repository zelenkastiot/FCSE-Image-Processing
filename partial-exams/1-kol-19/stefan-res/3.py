import cv2
import numpy as np

image = cv2.imread('Figure3.jpg', cv2.IMREAD_COLOR)

# red = [0, 255, 255]
lower_red = np.array([-10,100,100])
upper_red = np.array([10,255,255])

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(image, image, mask= mask)

cv2.imshow('result', res)
cv2.waitKey(0)
cv2.destroyAllWindows()