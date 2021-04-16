# max_rgb_filter is very useful tool to use when visualizing the Red, Green, and Blue channels
# of an image - and which channel contributes most to a given area of an image

# USAGE
# python max_filter.py --image images/horseshoe_bend_02.jpg

# import the necessary packages
import numpy as np
import argparse
import cv2

def max_rgb_filter(image):
    # split the image into its BGR components
    (B, G, R) = cv2.split(image)

    # find the maximum pixel intensity values for each
    # (x, y)-coordinate, then set all pixel values less
    # than M to zero
    M = np.maximum(np.maximum(R, G), B)
    # use np.maximum and not np.max! The np.max  method will only find the
    # maximum value across the entire array as opposed to np.maximum which
    # find the max value at each (x, y)-coordinate.
    R[R < M] = 0
    G[G < M] = 0
    B[B < M] = 0

    # merge the channels back together and return the image
    return cv2.merge([B, G, R])


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
args = vars(ap.parse_args())

# load the image, apply the max RGB filter, and show the
# output images
image = cv2.imread(args["image"])
filtered = max_rgb_filter(image)
cv2.imshow("Images", np.hstack([image, filtered]))
cv2.waitKey(0)
