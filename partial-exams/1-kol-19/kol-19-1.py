"""

 Created on 15-Apr-21
 @author: Kiril Zelenkovski

"""
import cv2
import matplotlib.pyplot as plt

plt.rcParams['savefig.facecolor'] = "0.8"
img = cv2.imread("Figure1.jpg", 0)

# Obicen hist
plt.subplot(3, 2, 1)
plt.title("Original image")
plt.imshow(img, cmap='gray')

plt.subplot(3, 2, 2)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.hist(img.ravel(), 256, [0, 256], color="darkred")

# # Hist poramnet
plt.subplot(3, 2, 3)
plt.title("Global histogram eq.")
eqImg = cv2.equalizeHist(img)
plt.imshow(eqImg, cmap='gray')


plt.subplot(3, 2, 4)
plt.hist(eqImg.ravel(), 256, [0, 256], color="darkred")

# # Adaptive poramnuvanje
plt.subplot(3, 2, 5)
plt.title("Adaptive histogram eq.")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
plt.imshow(cl1, cmap='gray')

plt.subplot(3, 2, 6)
plt.hist(cl1.ravel(), 256, [0, 256], color="darkred")

plt.tight_layout()
plt.savefig("rezultat-1.jpg")
plt.show()
