import cv2
from matplotlib import pyplot as plt

# Load test image
img = cv2.imread('Noise_salt_and_pepper.png')
# Gaussian kernel
median = cv2.medianBlur(img, 5)
#median = cv2.GaussianBlur(img,(5,5),0)
# Show images
plt.figure(1)
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(median)
plt.show()
# Save images
cv2.imwrite('median-img-1.png', median)
