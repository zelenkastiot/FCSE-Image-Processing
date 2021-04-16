import cv2

img = cv2.imread('messi5.jpg', 0)
cv2.imwrite('messigray.png', img)
