import cv2
import numpy as np
# generate Gaussian pyramid for Apple and Orange
# generate Laplacian Pyramids for Gaussian pyramids
# Now add left and right halves of images in each level of the Laplacian Pyramids
# From the join image pyramids reconstruct the original image

apple = cv2.imread('apple.png')
orange = cv2.imread('orange.png')
rows_apple, columns_apple, channels_apple = apple.shape
rows_orange, columns_orange, channels_orange = orange.shape

print(apple.shape)
print(orange.shape)

apple_orange = np.hstack((apple[:,0:int(columns_apple/2)],orange[:,int(columns_orange/2):]))
cv2.imshow('Result',apple_orange)
cv2.waitKey(0)

# generate Gaussian pyramid for apple
apple_copy = apple.copy()
gp_apple = [apple_copy]
for i in range(6):
    apple_copy = cv2.pyrDown(apple_copy)
    gp_apple.append(apple_copy)

# generate Gaussian pyramid for orange
orange_copy = orange.copy()
gp_orange = [orange_copy]
for i in range(6):
    orange_copy = cv2.pyrDown(orange_copy)
    gp_orange.append(orange_copy)

# generate Laplacian pyramid for apple
apple_copy = gp_apple[5]
lp_apple = [apple_copy]
for i in range(5,0,-1):
    gaussian_expanded = cv2.pyrUp(gp_apple[i])
    laplacian = cv2.subtract(gp_apple[i-1],gaussian_expanded)
    lp_apple.append(laplacian)

# generate Laplacian pyramid for orange
orange_copy = gp_orange[5]
lp_orange = [orange_copy]
for i in range(5,0,-1):
    gaussian_expanded = cv2.pyrUp(gp_orange[i])
    laplacian = cv2.subtract(gp_orange[i-1],gaussian_expanded)
    lp_orange.append(laplacian)

# Now add left and right halves of images in each level
apple_orange_pyramid = []
for apple_laplacian,orange_laplacian in zip(lp_apple,lp_orange):
    rows,cols,dpt = apple_laplacian.shape
    laplacian = np.hstack((apple_laplacian[:,0:int(cols/2)], orange_laplacian[:,int(cols/2):]))
    apple_orange_pyramid.append(laplacian)

# now reconstruct
apple_ornage_reconstruct = apple_orange_pyramid[0]
cv2.waitKey(0)
for i in range(1,6):
    apple_ornage_reconstruct = cv2.pyrUp(apple_ornage_reconstruct)
    apple_ornage_reconstruct = cv2.add(apple_ornage_reconstruct, apple_orange_pyramid[i])

cv2.imshow('Result',apple_ornage_reconstruct)
cv2.waitKey(0)