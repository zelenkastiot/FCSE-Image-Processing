import cv2
import numpy as np

image = cv2.imread('Figure4.jpg', cv2.IMREAD_GRAYSCALE)
image = np.float32(image)

prewitt_x = np.array(([-1, -1, -1], [0, 0, 0], [1, 1, 1]), dtype="float32")
prewitt_y = np.array(([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]), dtype="float32")

edges_x = cv2.filter2D(image, -1, prewitt_x)
edges_x = np.abs(edges_x)

edges_y = cv2.filter2D(image, -1, prewitt_y)
edges_y = np.abs(edges_y)

edges = np.sqrt(edges_x**2, edges_y**2)
edges = np.uint8(edges)

# otsu samiot go naoga optimalniot prag
ret3,th3 = cv2.threshold(edges,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('edges', th3)
cv2.waitKey(0)
cv2.destroyAllWindows()