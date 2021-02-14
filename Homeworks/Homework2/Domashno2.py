import numpy as np
import cv2
import matplotlib.pyplot as plt


def make_filtered_image(image, filter):
	filtered_img = cv2.filter2D(image, -1, filter)
	filtered_img = abs(filtered_img)
	return filtered_img / np.amax(filtered_img[:])


img = cv2.imread("Lenna.png", 0)

n = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype='float32')
ne = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]], dtype='float32')
e = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype='float32')
se = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype='float32')
s = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype='float32')
sw = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]], dtype='float32')
w = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype='float32')
nw = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]], dtype='float32')

filtered = []
filters = [n, ne, e, se, s, sw, w, nw]
labels = ["North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest"]
# apply all filters on the image
for filter in filters:
	filtered.append(make_filtered_image(img, filter))

# plot the filtered image based on direction
for i, f in enumerate(filtered):
	plt.subplot(2, 4, i + 1)
	plt.imshow(f, cmap='gray')
	plt.title(f"Direction: {labels[i]}")

plt.subplots_adjust(left=0.033, bottom=0.008, right=0.942, top=0.982, wspace=0.333, hspace=0.000)
plt.tight_layout()
plt.show()

# calculate total edge sum
edge_total = np.maximum.reduce(filtered)

plt.subplot(131)
plt.imshow(img, cmap='gray')
plt.title("Original image (gray scaled)")

# plot original picture and the summed one
plt.subplot(132)
plt.imshow(edge_total, cmap='gray')
plt.title("      Combined total (no threshold)")

gaussian = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,17,6.5)
plt.subplot(133)
plt.imshow(gaussian, cmap='gray')
plt.title("Mean threshold")
plt.tight_layout()
plt.show()

# apply threshold and plot the filtered image
for i, j in zip(np.arange(0.05, 1, 0.05), range(0, 19)):
	plt.subplot(4, 5, j + 1)
	_, thresh = cv2.threshold(edge_total, i, 1, cv2.THRESH_BINARY)
	plt.imshow(thresh, cmap='gray')
	plt.title(f"Threshold: {round(i, 2)}")
plt.subplots_adjust(left=0.031, bottom=0.026, right=0.961, top=0.961, wspace=0.365, hspace=0.270)

plt.tight_layout()
plt.show()