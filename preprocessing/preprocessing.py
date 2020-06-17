""" Perona Malik Diffusion
    for Edge Detection
"""

import numpy as np
from scipy import misc, ndimage
import matplotlib.pyplot as plt
import cv2


# SETTINGS:
image_file = 'noisy.tif'
iterations = 20
delta = 0.14
kappa = 8

# convert input image
im = misc.imread(image_file, flatten=True)
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



fig = plt.figure(frameon=False)
fig.set_size_inches(5, 5)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

# img = cv2.medianBlur(u.astype('uint8'),5)
#
# th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
#             cv2.THRESH_BINARY,31,10)
# plt.imshow(img,'gray')


# img_grey = cv2.cvtColor(th3, cv2.COLOR_BGR2GRAY)
#
# size = np.size(img_grey)
# skel = np.zeros(img_grey.shape, np.uint8)
#
# ret, img = cv2.threshold(img_grey, 127, 255, 0)
# element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# done = False
#
# while (not done):
#     eroded = cv2.erode(img, element)
#     temp = cv2.dilate(eroded, element)
#     temp = cv2.subtract(img, temp)
#     skel = cv2.bitwise_or(skel, temp)
#     img = eroded.copy()
#
#     zeros = size - cv2.countNonZero(img)
#     if zeros == size:
#         done = True
#
# plt.imshow("skel", skel)


#plt.imshow(median, cmap='gray')
plt.show()
#plt.savefig("./filtered.png")


#show all
# plt.subplot(1, 3, 1), plt.imshow(im, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.xlabel('Original')
# plt.subplot(1, 3, 2), plt.imshow(u, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.xlabel('After diffusion')
# plt.subplot(1, 3, 3 ), plt.imshow(G, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.xlabel('Gradient after diffusion')
# plt.show()
