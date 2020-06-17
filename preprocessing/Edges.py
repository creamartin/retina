import cv2
import numpy as np

img = cv2.imread('test_sobel.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)

sobely.max
cv2.imshow("Sobel Image", sobely)
cv2.imshow("Original Image", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
