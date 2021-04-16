import numpy as np
import cv2

img = cv2.imread('image2.jpg')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# the unspecified value is inferred to be number of pixels
Z = img.reshape((-1, 3))
print(Z)
# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 10
compactness, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
print(center)
print(label)
res = center[label.flatten()]
print(res)
res2 = res.reshape((img.shape))
#res2 = cv2.cvtColor(res2, cv2.COLOR_LAB2BGR)

cv2.imshow('res2', res2)
cv2.imwrite('img_rgb.jpg', res2)
# cv2.imwrite('img_lab.jpg', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
