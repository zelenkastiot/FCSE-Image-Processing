import cv2

image = cv2.imread('Figure2.jpg', cv2.IMREAD_COLOR)
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
(h, s, v) = cv2.split(image_hsv)

cv2.imshow('hue', h)

cv2.imshow('saturation', s)

cv2.imshow('value', v)

cv2.waitKey(0)
cv2.destroyAllWindows()