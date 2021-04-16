import numpy as np
import cv2
from matplotlib import pyplot as plt

# Return a random integer N such that a <= N <= b
x = np.random.randint(25, 100, (25, 1))
y = np.random.randint(175, 255, (25, 1))
print(y)
z = np.vstack((x, y))
# convert to np.float32
z = np.float32(z)
plt.hist(z, 256, [0, 256])
plt.show()
# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
compactness, labels, centers = cv2.kmeans(z, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
print('labels: {}'.format(labels))
print('centers: {}'.format(centers))
# Now separate the data
A = z[labels.ravel() == 0]
B = z[labels.ravel() == 1]
# Plot the data
plt.hist(A, 256, [0, 256], color='r')
plt.hist(B, 256, [0, 256], color='b')
plt.hist(centers, 32, [0, 256], color='y')
plt.show()
