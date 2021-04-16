import cv2

img = cv2.imread('image1.jpg')
print(img)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray_img)
cv2.imshow('Grayscale image', gray_img)
cv2.waitKey()
