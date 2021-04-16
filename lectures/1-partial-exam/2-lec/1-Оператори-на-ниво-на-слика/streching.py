import cv2
import numpy as np

# Function to map each intensity level to output intensity level.
def pixelVal(pix, r1, s1, r2, s2):
    if 0 <= pix and pix <= r1:
        return (s1 / r1) * pix
    elif r1 < pix and pix <= r2:
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2


img = cv2.imread('sample.jpg', 0)

# Define parameters.
r1 = 70.0
s1 = 0.0
r2 = 140.0
s2 = 255.0

# Vectorize the function to apply it to each value in the Numpy array.
pixelVal_vec = np.vectorize(pixelVal)

# Apply contrast stretching.
contrast_stretched = pixelVal_vec(img, r1, s1, r2, s2)
contrast_stretched = np.array(contrast_stretched, dtype=np.uint8)
cv2.imshow('contrast_stretched', contrast_stretched)
cv2.waitKey(0)
cv2.destroyAllWindows()