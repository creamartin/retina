import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:/Users/ylliv/Documents/Medientechnologie/Medientechnologie/6.Semester/Retina/Python/437D6CD0.tif',0)
blur = cv2.blur(img,(5,5))
gblur = cv2.GaussianBlur(img,(5,5),0)
bilblur = cv2.bilateralFilter(img,9,25,25)
#anis = cv2.ximgproc.anisotropicDiffusion (img, 0.5, 0.3, 5)

plt.subplot(221),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(gblur),plt.title('Gaussian')
plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(bilblur),plt.title('Bilateral')
plt.xticks([]), plt.yticks([])
plt.show()
