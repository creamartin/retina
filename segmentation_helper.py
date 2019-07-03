import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from skimage.filters import threshold_otsu


# cropped,top,bottom = find_roi(img)
def find_roi(img):
    img = img.astype('float32')

    # binarize
    thresh = threshold_otsu(img)
    binary = img > thresh
    binary = binary.astype(np.uint8)

    # close
    kernel = np.ones((25, 25), np.uint8)
    close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # crop
    height, width = close.shape
    first_white = int(np.argmax(close > 0) / width)
    last_white = height - int(np.argmax(np.flip(close) > 0) / width)
    crop = img[:last_white, :]
    return crop, first_white, last_white


def flatten(img):
    img = img.astype('float32')

    # denoise Gaussian
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    gaussian = cv2.medianBlur(img, 5, 0)

    # tentatively assign brightest pixel as RPE
    width, height = gaussian.shape
    indicesX = np.array([], dtype=np.int8)
    indicesY = np.array([], dtype=np.int8)

    snrs = np.array([])

    for i in range(width):
        cur_col = np.array(gaussian[:, i])
        snr = signaltonoise(cur_col, axis=0, ddof=0)
        # not sure about 0.7?
        if snr > 0.7:
            maxi = np.where(cur_col == np.amax(cur_col))[0][0]
            indicesX = np.append(indicesX, i)
            indicesY = np.append(indicesY, maxi)
            snrs = np.append(snrs, snr)
    # remove outliers greater 50pixels
    median = np.mean(indicesY)
    print(median)
    newX = np.array([],dtype=np.int8)
    newY = np.array([],dtype=np.int8)
    for i in range(len(indicesY)):
        el = indicesY[i]
        if el < median + 60 and el > median - 60:
            newX = np.append(newX,indicesX[i])
            newY = np.append(newY,el)
    indicesX = newX
    indicesY = newY
    # mean = np.mean(indices[:][0])
    # print("mean: "+str(mean)+"\nmedian:"+str(median))
    #indicesY = [el for el in indicesY if el < median + 25 or el > median - 25]

    # remove columns that present a significantly lower signal-to-noise ratio

    # fit 2nd order polynomial to RPE Points
    # shift each column up or down such that the RPE points lie on a flat line
    backtorgb = cv2.cvtColor(gaussian, cv2.COLOR_GRAY2RGB).copy()
    print(len(indicesX))
    print(len(indicesY))

    print(len(backtorgb))

    for i in range(len(indicesX)):
        x = indicesX[i]
        y = indicesY[i]
        backtorgb[y, x] = (0, 0, 255)

    return backtorgb


def signaltonoise(a, axis, ddof):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


# examples
def exampleCrop():
    image_file = '../testvectors/sick.tif'
    im = io.imread(image_file, as_gray=True)
    im = im.astype('float32')
    cropped, top, bottom = find_roi(im)
    plt.gca().imshow(cropped, cmap='gray')
    print(top, bottom)
    plt.figure()
    plt.gca().imshow(im, cmap='gray')
    plt.setp(plt.gca(), xticks=[], yticks=[])
    plt.show()


def exampleFlatten():
    image_file = '../testvectors/ref.tif'
    im = io.imread(image_file, as_gray=True)
    im = im.astype('float32')
    flattened = flatten(im)
    plt.gca().imshow(flattened)
    #plt.figure()
    #plt.gca().imshow(im, cmap='gray')
    plt.setp(plt.gca(), xticks=[], yticks=[])
    plt.show()


exampleFlatten()
