import cv2

x=cv2.imread('zelda.pgm', 0)
cv2.imshow('image', x)
cv2.waitKey(0)

#digital negative
y=255-x
cv2.imshow('image negative', y)
cv2.waitKey(0)
cv2.destroyAllWindows()