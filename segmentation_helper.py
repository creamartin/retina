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

# flattened, unshiftList = flatten(img)
# unflattened = shiftColumn(flattened, unshiftList)
def flatten(img):
    img = img.astype('float32')
    # denoise Gaussian
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)

    # tentatively assign brightest pixel as RPE
    height, width = gaussian.shape
    indicesX = np.array([], dtype=np.int8)
    indicesY = np.array([], dtype=np.int8)
    snrs = np.array([], dtype=np.float)
    for i in range(width):
        cur_col = np.array(gaussian[:, i])
        snr = signaltonoise(cur_col, axis=0, ddof=0)
        maxi = np.where(cur_col == np.amax(cur_col))[0][0]
        indicesX = np.append(indicesX, i)
        indicesY = np.append(indicesY, maxi)
        snrs = np.append(snrs, snr)

    # remove columns that present a significantly lower signal-to-noise ratio
    mean = np.mean(snrs)
    # 5% less than average snr allowed
    limit = mean - 0.05 * mean
    remove = np.where(snrs < limit)[0]
    indicesX = np.delete(indicesX, remove)
    indicesY = np.delete(indicesY, remove)

    # remove outliers greater 60pixels
    median = np.median(indicesY)
    newX = np.array([], dtype=np.int8)
    newY = np.array([], dtype=np.int8)
    for i in range(len(indicesY)):
        el = indicesY[i]
        if el < median + 60 and el > median - 60:
            newX = np.append(newX, indicesX[i])
            newY = np.append(newY, el)
    indicesX = newX
    indicesY = newY

    # fit 2nd order polynomial to RPE Points
    coeffs = np.polyfit(indicesX, indicesY, 3)
    approximation = np.poly1d(coeffs)

    # #draw line
    # backtorgb = cv2.cvtColor(gaussian, cv2.COLOR_GRAY2RGB)
    # print(backtorgb.shape)
    # for i in range(width):
    #     backtorgb[int(approximation(i)), i] = (255, 0, 0)
    #
    # return backtorgb,[]

    # shift each column up or down such that the RPE points lie on a flat line
    shiftList = np.array([], dtype=np.int8)

    for i in range(width):
        shiftList = np.append(shiftList, int(approximation(i)))
    maxShift = max(shiftList)
    shiftList = [int(maxShift-x) for x in shiftList]
    print(shiftList)

    return shiftColumn(shiftList, gaussian.copy()), [-x for x in shiftList]

def shiftColumn (shift_list, img):
    height, width = img.shape

    for i in range(len(shift_list)):
        tmp_column = img[:, i]
        shift = int(shift_list[i])
        if shift > 0:
            img[shift:height - 1, i] = tmp_column[0:height - 1 - shift]
            img[0:shift - 1, i] = np.flip(tmp_column[:shift - 1])
        if shift < 0:
            img[0:height - 1 + shift, i] = tmp_column[abs(shift):height - 1]
            img[height -1 +shift:height-1, i] = np.flip(tmp_column[height -1 +shift:height-1])

    return img

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
    image_file = '../testvectors/sick.tif'
    im = io.imread(image_file, as_gray=True)
    im = im.astype('float32')
    flattened, shiftlist = flatten(im)
    plt.gca().imshow(flattened, cmap='gray')
    plt.setp(plt.gca(), xticks=[], yticks=[])
    plt.show()

