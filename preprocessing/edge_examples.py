""" Perona Malik Diffusion
    for Edge Detection
"""

import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt
import cv2
from skimage.filters import roberts, sobel, sobel_h, sobel_v, scharr, \
    scharr_h, scharr_v, prewitt, prewitt_v, prewitt_h


# SETTINGS:
image_file = 'assets/testvectors/sick.tif'
iterations = 2
delta = 0.14
kappa = 6

# convert input image
im = misc.imread(image_file, flatten=True)
#img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#

#img1 = np.uint8(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX))
#clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
#adaptiveHis = clahe.apply(img1)
#median = cv2.medianBlur(adaptiveHis,1)
#plt.imshow(adaptiveHis,cmap=plt.cm.gray)
#plt.show()

im = im.astype('float64')

# initial condition
u = im

# center pixel distances
dx = 1
dy = 1
dd = np.sqrt(2)

# 2D finite difference windows
windows = [
    np.array(
            [[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.float64
    ),
    np.array(
            [[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float64
    ),
    np.array(
            [[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float64
    ),
    np.array(
            [[0, 0, 0], [1, -1, 0], [0, 0, 0]], np.float64
    ),
    np.array(
            [[0, 0, 1], [0, -1, 0], [0, 0, 0]], np.float64
    ),
    np.array(
            [[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64
    ),
    np.array(
            [[0, 0, 0], [0, -1, 0], [1, 0, 0]], np.float64
    ),
    np.array(
            [[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.float64
    ),
]

for r in range(iterations):
    # approximate gradients
    nabla = [ ndimage.filters.convolve(u, w) for w in windows ]

    # approximate diffusion function
    diff = [ 1./(1 + (n/kappa)**2) for n in nabla]

    # update image
    terms = [diff[i]*nabla[i] for i in range(4)]
    terms += [(1/(dd**2))*diff[i]*nabla[i] for i in range(4, 8)]
    u = u + delta*(sum(terms))


# Kernel for Gradient in x-direction
Kx = np.array(
    [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32
)
#Kernel for Gradient in y-direction
Ky = np.array(
    [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32
)
# Apply kernels to the image
Ix = ndimage.filters.convolve(u, Kx)
Iy = ndimage.filters.convolve(u, Ky)

# return norm of (Ix, Iy)
G = np.hypot(Ix, Iy)
#G = ndimage.filters.median_filter(G,5)


gray = u.astype(np.uint8)
print(gray.dtype)
img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

#canny
img_canny = cv2.Canny(gray,100,255)

#sobel
img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
img_sobel = img_sobelx + img_sobely

#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)
img_prewitt = img_prewitty + img_prewittx


size = np.size(img_canny)
skel = np.zeros(img_canny.shape,np.uint8)

ret,img = cv2.threshold(img_canny,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False

while( not done):
    eroded = cv2.erode(img,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(img,temp)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy()

    zeros = size - cv2.countNonZero(img)
    if zeros==size:
        done = True

closing = cv2.morphologyEx(img_canny, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))


# fig = plt.figure(frameon=False)
# fig.set_size_inches(5, 5)
# ax = plt.Axes(fig, [0., 0., 1., 1.])
# ax.set_axis_off()
# fig.add_axes(ax)
# #plt.show()
# #plt.savefig("./filtered.png")


plt.subplot(2, 3, 1), plt.imshow(im, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.xlabel('Original')
plt.subplot(2, 3, 2), plt.imshow(u, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.xlabel('After diffusion')
plt.subplot(2, 3, 3 ), plt.imshow(G, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.xlabel('Gradient after diffusion')
plt.subplot(2, 3, 4 ), plt.imshow(img_prewitt, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.xlabel('Prewitt')
plt.subplot(2, 3, 5 ), plt.imshow(img_sobel, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.xlabel('Sobel')

plt.figure(frameon=False)
plt.imshow(skel, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.xlabel('Canny')
plt.tight_layout();

edge_sobel = sobel(u)
edge_scharr = scharr(u)
edge_prewitt = prewitt(u)
plt.figure(frameon=False)
plt.imshow(edge_sobel, cmap='gray')
plt.figure(frameon=False)
plt.imshow(edge_scharr, cmap='gray')
plt.figure(frameon=False)
plt.imshow(edge_prewitt, cmap='gray')

plt.show()


