import numpy as np
import cv2
import os
import pickle
import matplotlib.pyplot as plt
from pyimagesearch.panorama import Stitcher
from functools import reduce

def write_to_pickle(dir_name, index):
	with open(f"{dir_name}.cpickle", "wb") as f:
		f.write(pickle.dumps(index))
		f.close()


def write_to_text(dir_name, index):
	with open(f"{dir_name}.txt", "w") as f:
		f.write(str(index))
		f.close()


# calculate the keypoints and descriptors for each image in the dataset
def get_keypoints_database(dir_name):
	index = {}
	for img_name in os.listdir(dir_name):
		pts = []
		img = cv2.imread(f"{dir_name}/{img_name}", 0)
		# we rescale it because it takes a lot of memory to process the images
		img = cv2.resize(img,(800,600))
		sift = cv2.xfeatures2d.SIFT_create()
		(kps, descs) = sift.detectAndCompute(img, None)
		for point in kps:
			temp = (*point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
			pts.append(temp)

		index[f"{dir_name}/{img_name}"] = (pts, descs)

	write_to_pickle(dir_name, index)
	write_to_text(dir_name, index)

	return index

def solid(list):
	s = 0
	for l in list:
		s += sum(l)
	return s


dirname = "Database"
queryname = "Queries"
exists = os.path.isfile(f"{dirname}.cpickle")
if exists:
	desc = pickle.loads(open(f"{dirname}.cpickle","rb").read())
else:
	desc = get_keypoints_database(dir_name=dirname)

# convert keypoints back to cv2.KeyPoint objects
for k, v in desc.items():
	kp = []
	for point in v[0]:
		kpp = cv2.KeyPoint(x=point[0],y=point[1],_size=point[2], _angle=point[3], _response=point[4], _octave=point[5], _class_id=point[6])
		kp.append(kpp)
	desc[k] = (kp, v[1])

dd = {}
for img_name in os.listdir(queryname):
	best = []
	length = -np.inf
	path = ""
	bmatches = None
	keypoints = []
	img = cv2.imread(f"{queryname}/{img_name}")
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	(kps, descs) = sift.detectAndCompute(gray, None)
	# FLANN parameters
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=10)
	search_params = dict(checks=55)  # or pass empty dictionary

	flann = cv2.FlannBasedMatcher(index_params, search_params)
	for k, v in desc.items():
		matches = flann.knnMatch(descs, v[1], k=2)
		good = [[0,0] for i in range(len(matches))]
		for i, (m, n) in enumerate(matches):
			if m.distance < 0.7 * n.distance:
				good[i] = [1, 0]
		ll = solid(good)
		if ll >= length:
			length = ll
			path = k
			best = good[:]
			keypoints = v[0]
			bmatches = matches

	dd[f"{queryname}/{img_name}"] = (best, path, keypoints)
	draw_params = dict(singlePointColor = (255,255,0), matchColor=(255, 255, 0) , matchesMask=best, flags=2)
	img2 = cv2.resize(cv2.imread(path),(800, 600))
	img3 = cv2.drawMatchesKnn(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), kps, cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), keypoints, bmatches, None,**draw_params)
	plt.imshow(img3), plt.show()

# after we find best matched images, we do the RANSAC algorithm
for k, v in dd.items():
	imageA = cv2.resize(cv2.imread(k), (800, 600))
	stitcher = Stitcher()
	imageB = cv2.resize(cv2.imread(v[1]), (800, 600))

	(result, vis) = stitcher.stitch([imageB, imageA], showMatches=True)
	# show the images
	cv2.imshow("Keypoint Matches", vis)
	cv2.waitKey(0)w