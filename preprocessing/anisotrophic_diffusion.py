""" Perona Malik Diffusion
    for Edge Detection
"""

import numpy as np
from matplotlib import gridspec
from scipy import misc, ndimage
import matplotlib.pyplot as plt
import cv2
import skimage.io as io

image_file = 'assets/testvectors/sick.tif'
img = io.imread(image_file, as_gray=True)


def anisotropic_diffusion(im, i, d, k):
    _im = im.astype('float32')
    # initial condition
    u = _im

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
        nabla = [ndimage.filters.convolve(u, w) for w in windows]

        # approximate diffusion function
        diff = [1. / (1 + (n / kappa) ** 2) for n in nabla]

        # update image
        terms = [diff[i] * nabla[i] for i in range(4)]
        terms += [(1 / (dd ** 2)) * diff[i] * nabla[i] for i in range(4, 8)]
        u = u + delta * (sum(terms))

    return u


def gradient_sobel(im):
    # Kernel for Gradient in x-direction
    Kx = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32
    )
    # Kernel for Gradient in y-direction
    Ky = np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32
    )
    # Apply kernels to the image
    Ix = ndimage.filters.convolve(im, Kx)
    Iy = ndimage.filters.convolve(im, Ky)

    # return norm of (Ix, Iy)
    G = np.hypot(Ix, Iy)

    return G


for _i in range(3):
    # fig, axs = plt.subplots(3, 9, frameon=False, figsize=(9, 4), sharex=True, sharey=True)
    #

    fig = plt.figure("Iterations: " + str(_i * 10 + 5), frameon=False, figsize=(9, 3))

    gs = gridspec.GridSpec(3, 9, wspace=0.0, hspace=0.0)

    for _d in range(3):
        for _k in range(9):
            iterations = 5 + _i * 10
            delta = 0.1 + 0.1 * float(_d)
            kappa = 1 + _k

            diffused = anisotropic_diffusion(img, iterations, delta, kappa)

            axis = fig.add_subplot(gs[_d, _k])
            plt.xticks(fontsize=6)
            plt.yticks(fontsize=6)
            plt.setp(axis, xticks=[], yticks=[])
            axis.imshow(diffused, cmap='gray')
            axis.set_aspect('equal')
            if _k == 0:
                axis.set(ylabel='d: ' + str(round(delta,2)))
            if _d == 2:
                axis.set(xlabel='k:' + str(kappa))
#plt.subplots_adjust(wspace=0, hspace=0)
#plt.autoscale(tight=True)
plt.savefig("./filtered.png",dpi=400)
#plt.show()
