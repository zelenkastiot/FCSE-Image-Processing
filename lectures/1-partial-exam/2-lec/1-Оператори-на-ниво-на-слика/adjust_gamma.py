#gamma correction, a translation between the sensitivity of our eyes and sensors of a camera.
#Gamma correction is also known as the Power Law Transform.
# First, our image pixel intensities must be scaled from the range [0, 255] to [0, 1.0].
# From there, we obtain our output gamma corrected image by applying the following equation:
#O = I ^ (1 / G)
#Where I is our input image and G is our gamma value.
# The output image O is then scaled back to the range [0, 255].
#Gamma values < 1 will shift the image towards the darker end of the spectrum
# while gamma values > 1 will make the image appear lighter.
# A gamma value of G=1 will have no affect on the input image.

# USAGE
# python adjust_gamma.py --image image1.png

# import the necessary packages
import numpy as np
import argparse
import cv2


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# load the original image
original = cv2.imread(args["image"])

# loop over various values of gamma
for gamma in np.arange(0.0, 3.5, 0.5):
	# ignore when gamma is 1 (there will be no change to the image)
	if gamma == 1:
		continue

	# apply gamma correction and show the images
	gamma = gamma if gamma > 0 else 0.1
	adjusted = adjust_gamma(original, gamma=gamma)
	cv2.putText(adjusted, "g={}".format(gamma), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
	cv2.imshow("Images", np.hstack([original, adjusted]))
	cv2.waitKey(0)
