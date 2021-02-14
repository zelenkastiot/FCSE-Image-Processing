import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
from scipy.spatial.distance import minkowski, euclidean


class Searcher:
	def __init__(self, index):
		# store our index of images
		self.index = index

	def search(self, queryFeatures):
		# initialize our dictionary of results
		results = {}

		# loop over the index
		for (k, features) in self.index.items():
			# compute the distance between the query features
			# and features in our index, then update the results
			d = euclidean(features, queryFeatures)

			# now that we have the distance between the two feature
			# vectors, we can udpate the results dictionary -- the
			# key is the current image ID in the index and the
			# value is the distance we just computed, representing
			# how 'similar' the image in the index is to our query
			results[k] = d

		# sort our results, so that the smaller distances (i.e. the
		# more relevant images are at the front of the list)
		results = sorted([(v, k) for (k, v) in results.items()])
		print(results)

		# return our results
		return results

	def chi2_distance(self, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
			for (a, b) in zip(histA, histB)])

		# return the chi-squared distance
		return d



database = pickle.loads(open("database.cpickle","rb").read())
queries = pickle.loads(open("query.cpickle", "rb").read())

searcher = Searcher(database)
select_top = 3
for name, query_features in queries.items():
	similar = searcher.search(query_features)[:select_top]
	plt.imshow(cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB))
	plt.title("Query image: "+name)
	plt.show()
	for i, sim in enumerate(similar):
		plt.subplot(1, select_top, i+1)
		plt.imshow(cv2.cvtColor(cv2.imread(sim[1]), cv2.COLOR_BGR2RGB))
		plt.title(sim[1])

	plt.show()