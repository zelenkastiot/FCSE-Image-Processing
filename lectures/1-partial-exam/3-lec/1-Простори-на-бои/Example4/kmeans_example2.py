import numpy as np
import cv2
from matplotlib import pyplot as plt

x = np.random.randint(25, 50, (25, 2))
print(x)
y = np.random.randint(60, 85, (25, 2))
z = np.vstack((x, y))
# convert to np.float32
z = np.float32(z)
# define criteria and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
compactness, label, center = cv2.kmeans(z, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
print(z, label.ravel())
# Now separate the data, Note the flatten()
A = z[label.ravel() == 0]
B = z[label.ravel() == 1]

# Plot the data
plt.scatter(A[:, 0], A[:, 1])
plt.scatter(B[:, 0], B[:, 1], c='r')
plt.scatter(center[:, 0], center[:, 1], s=80, c='y', marker='s')
plt.xlabel('Height'), plt.ylabel('Weight')
plt.show()
