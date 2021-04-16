import cv2

img = cv2.imread('messi5.jpg', 1)
img[:,:,0] = 0
img[:,:,1] = 0
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
