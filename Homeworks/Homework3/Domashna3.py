import numpy as np
import cv2
import os
import mahotas
import pickle


def write_to_pickle(dir_name, index):
	with open(f"{dir_name}.cpickle", "wb") as f:
		f.write(pickle.dumps(index))
		f.close()


def write_to_text(dir_name, index):
	with open(f"{dir_name}.txt", "w") as f:
		f.write(str(index))
		f.close()


def calculate_features(dir_name):
	index = {}
	for file_path in os.listdir(dir_name):
		img = cv2.imread(f"{dir_name}/{file_path}", 0)
		Z = img.reshape((-1, 1))
		# convert to np.float32
		Z = np.float32(Z)

		# define criteria, number of clusters(K) and apply kmeans()
		# 100 means for one type of centroids, we adjust them for 100 iterations
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 110, 1.0)
		K = 2
		# 55 means generate 55 random centroids and find the best ones
		ret, label, center = cv2.kmeans(Z, K, None, criteria, 55, cv2.KMEANS_RANDOM_CENTERS)

		# Now convert back into uint8, and make original image
		center = np.uint8(center)
		res = center[label.flatten()]
		res2 = res.reshape((img.shape))

		# reducing the small white holes (erode then dilate)
		se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
		BW1 = cv2.erode(res2, se1, iterations=4)

		BW1 = cv2.dilate(BW1, se1, iterations=4)
		BW1 = cv2.bitwise_not(BW1)

		# get the contour points of image mask
		ret, thresh1 = cv2.threshold(BW1.copy(), 170, 255, cv2.THRESH_BINARY)
		contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		#print(contours[0])
		# if there are any contour points(more than 14 preferably), calculate the circle area of operation with radius and center (x, y)
		# else use some default value
		if len(contours) > 10:
			(x, y), radius = cv2.minEnclosingCircle(contours[0])
			mh = mahotas.features.zernike_moments(BW1, radius=radius, cm=(x,y))
		else:
			mh = mahotas.features.zernike_moments(BW1, radius=50)

		index[f"{dir_name}/{file_path}"] = mh

	# write the dictionary like pickle and txt files
	write_to_pickle(dir_name, index)
	write_to_text(dir_name, index)


calculate_features("database")
calculate_features("query")

print(cv2.__version__)