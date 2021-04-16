"""

 Created on 15-Apr-21
 @author: Kiril Zelenkovski

"""
import cv2

img = cv2.imread("Figure2.jpg", 1)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow("image - h", img_hsv[:, :, 0])
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("image - s", img_hsv[:, :, 1])
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("image - v", img_hsv[:, :, 2])
cv2.waitKey(0)
cv2.destroyAllWindows()
