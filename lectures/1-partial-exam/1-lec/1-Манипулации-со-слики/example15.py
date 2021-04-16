import cv2
from matplotlib import pyplot as plt

# Load two images
img1 = cv2.imread('messi5.jpg')
img2 = cv2.imread('opencv_logo.png')

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]

plt.figure(1)

# Now create a mask of logo and create its inverse mask also
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
plt.subplot(151)
plt.imshow(img2gray, cmap='gray')
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
plt.subplot(152)
plt.imshow(mask, cmap='gray')
mask_inv = cv2.bitwise_not(mask)
plt.subplot(153)
plt.imshow(mask_inv, cmap='gray')


# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
plt.subplot(154)
plt.imshow(img1_bg)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)
plt.subplot(155)
plt.imshow(img2_fg)
plt.show()
# Put logo in ROI and modify the main image
dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst

cv2.namedWindow('res', cv2.WINDOW_NORMAL)

cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
