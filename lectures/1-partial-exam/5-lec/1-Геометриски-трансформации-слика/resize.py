import cv2

img = cv2.imread('messi5.jpg')
cv2.imshow('Original image', img)

res1 = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
cv2.imshow('Resized image v1', res1)

height, width = img.shape[:2]
res2 = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
cv2.imshow('Resized image v2', res2)

cv2.waitKey(0)
cv2.destroyAllWindows()
