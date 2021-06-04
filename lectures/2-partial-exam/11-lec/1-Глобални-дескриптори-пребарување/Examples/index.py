# USAGE
# python index.py --dataset images --index index.pickle

# import the necessary packages
from pyimagesearch.rgbhistogram import RGBHistogram
import argparse
import pickle
import glob
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--index", required=True,
	help="Path to where the computed index will be stored")
args = vars(ap.parse_args())

# initialize the index dictionary to store our our quantifed
# images, with the 'key' of the dictionary being the image
# filename and the 'value' our computed features
index = {}

# initialize our image descriptor -- a 3D RGB histogram with
# 8 bins per channel
desc = RGBHistogram([8, 8, 8])

# use glob to grab the image paths and loop over them
for imagePath in glob.glob(args["dataset"] + "/*.png"):
	# load the image, describe it using our RGB histogram
	# descriptor, and update the index
	image = cv2.imread(imagePath)
	features = desc.describe(image)
	index[imagePath] = features

# we are now done indexing our image -- now we can write our
# index to disk
f = open(args["index"], "wb")

#The pickle module implements a fundamental, but powerful algorithm for serializing and de-serializing a Python object structure. 
#"Pickling" is the process whereby a Python object hierarchy is converted into a byte stream, and "unpickling" is the inverse operation, whereby a byte stream is converted back into an object hierarchy. 

f.write(pickle.dumps(index))
f.close()

# show how many images we indexed
print("done...indexed %d images" % (len(index)))