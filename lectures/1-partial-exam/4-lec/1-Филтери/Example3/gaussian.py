import cv2
from matplotlib import pyplot as plt

# Load test image
img = cv2.imread('croppedBike.png')
# Gaussian kernel
blur = cv2.GaussianBlur(img, (21, 21), 0)
# Show images
plt.figure(1)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(blur)
plt.show()
# Save images
cv2.imwrite('blur_1.png', blur)
