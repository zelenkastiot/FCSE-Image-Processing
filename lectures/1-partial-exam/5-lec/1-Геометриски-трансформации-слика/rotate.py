import cv2

img = cv2.imread('messi5.jpg', 0)
rows, cols = img.shape

# rotates the image by 90 degree with respect to center without any scaling
# cv2.getRotationMatrix2D(center, angle, scale)

M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
dst = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('img', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
