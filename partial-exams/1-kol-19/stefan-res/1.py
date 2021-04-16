import cv2
import matplotlib.pyplot as plt

image = cv2.imread('Figure1.jpg', cv2.IMREAD_GRAYSCALE)

plt.subplot(321)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(322)
plt.hist(image.ravel(), 256, [0,256])

equ = cv2.equalizeHist(image)
plt.subplot(323)
plt.imshow(equ, cmap='gray')
plt.title('Global histogram equalization')

plt.subplot(324)
plt.hist(equ.ravel(), 256, [0,256])

# regionot vrz koj se pravi ekvilizacija e 8x8 kolku pomal tolku podobar no pobavno
# clipLimit e koi vrednosti na pikseli se isfrleni od eq
plt.subplot(325)
clahe = cv2.createCLAHE(clipLimit=50.0, tileGridSize=(8,8))
cl1 = clahe.apply(image)
plt.imshow(image, cmap='gray')
plt.title('Adaptive histogram equalization')

plt.subplot(326)
plt.hist(cl1.ravel(),256,[0,256])

plt.show()