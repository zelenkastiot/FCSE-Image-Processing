import numpy as np
import cv2


def contrast_stretch(image, x: list, y: list):
	gray = False
	if len(image.shape) < 3:
		gray = True
	x.insert(0, 0)
	x.append(255)
	y.insert(0, 0)
	y.append(255)
	# calculate slope from points
	slopes = np.diff(y) / np.diff(x)
	dd = {}
	# combine ranges, slopes and points
	for i in range(len(x) - 1):
		# range1: (0, 50) range2: (51,150) etc
		dd[(x[i], x[i + 1]) if i == 0 else (x[i] + 1, x[i + 1])] = (slopes[i], y[i])

	if gray:
		# gray image has one channel
		image = stretch(image, dd)
	else:
		# make the stretching for all channels
		for i in range(image.shape[2]):
			image[:, :, i] = stretch(image[:, :, i], dd)

	return image


def stretch(channel, dd):
	masks = []
	for k in dd:
		# get all pixels with specified range as mask
		mask = cv2.inRange(channel, k[0], k[1])
		masks.append(mask)

	for vals, mask, x in zip(dd.values(), masks, dd.keys()):
		# get the pixels that we need to change
		px_to_modify = cv2.bitwise_and(channel, mask)
		# get the rest of the pixels
		px_help = cv2.bitwise_or(channel, mask)
		# create matrix rows x cols with values 255
		all_white = np.full((channel.shape[0], channel.shape[1]), 255, dtype=np.dtype('uint8'))
		non_mod_px_inv = cv2.bitwise_xor(px_help, all_white)

		# apply stretching to the chosen pixels
		px_to_modify[px_to_modify != 0] = (px_to_modify[px_to_modify != 0] - (x[0] - 1 if x[0] > 0 else x[0])) * vals[0] + vals[1]
		# add 255 to rest of the pixels (needed for xor)
		px_to_modify[px_to_modify == 0] = 255

		mod_vals = cv2.bitwise_xor(px_to_modify, non_mod_px_inv)
		channel = mod_vals

	return channel


img1 = cv2.imread("slika.jpg", 0)

x = [50, 150]
y = [30, 200]

cv2.imshow("smt", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

img1 = contrast_stretch(img1, x, y)

cv2.imshow("after", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()