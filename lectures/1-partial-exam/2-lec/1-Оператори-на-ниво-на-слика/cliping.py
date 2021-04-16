import cv2
import numpy as np

x = cv2.imread('zelda.pgm', 0)
cv2.imshow('image', x)
cv2.waitKey(0)

# clipping
orig_size = x.shape
flat_x = x.flatten()
# Given an interval, values outside the interval are clipped to the interval edges.
y = np.clip(flat_x, 50, 150).reshape(orig_size)
cv2.imshow('clipping', y)
cv2.waitKey(0)
cv2.destroyAllWindows()
