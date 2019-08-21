import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from scipy.signal import savgol_filter


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
            # img[el, indicesX[i]] = 1.
    indicesX = newX
    indicesY = newY

    # # fit 2nd order polynomial to RPE Points
    # coeffs = np.polyfit(indicesX, indicesY, 4)
    # approximation = np.poly1d(coeffs)
    # rpe = scipy.interpolate.UnivariateSpline(indicesX, indicesY)
    # # approximation.set_smoothing_factor(150)

    # alternatively
    nearest_neighbour_rpe = np.array([indicesY[find_nearest_return_index(indicesX, x)] for x in range(width)])
    nearest_neighbour_rpe = medfilt(nearest_neighbour_rpe, 3)
    nearest_neighbour_rpe = savgol_filter(nearest_neighbour_rpe, 51, 3)

    # shift each column up or down such that the RPE points lie on a flat line
    shiftList = np.array([], dtype=np.int8)

    for i in range(width):
        shiftList = np.append(shiftList, nearest_neighbour_rpe[i])
        # print rpe-line-estimate
        # img[int(nearest_neighbour_rpe[i]), i] = 1.
        # Todo is this okay?
        # col = img[:, i]
        # for y in range(height):
        #     if y > approx+20.:
        #         col[y] = 0.
        # img[:, i] = col
    maxShift = max(shiftList)
    shiftList = [int(maxShift - x) for x in shiftList]
    img = shiftColumn(shiftList, img)
    return img, [-x for x in shiftList]


def shiftColumn(shift_list, img):
    img = img.copy()
    if len(img.shape) > 2:
        height, width, channels = img.shape
        for color in range(4):
            for i in range(len(shift_list)):
                tmp_column = img[:, i, color]
                shift = int(shift_list[i])
                if shift > 0:
                    img[shift:height - 1, i, color] = tmp_column[0:height - 1 - shift]
                    img[0:shift - 1, i, color] = np.flip(tmp_column[:shift - 1])
                if shift < 0:
                    img[0:height - 1 + shift, i, color] = tmp_column[abs(shift):height - 1]
                    img[height - 1 + shift:height - 1, i, color] = np.flip(tmp_column[height - 1 + shift:height - 1])

    elif len(img.shape) == 2:
        height, width = img.shape
        for i in range(len(shift_list)):
            tmp_column = img[:, i]
            shift = int(shift_list[i])
            if shift > 0:
                img[shift:height - 1, i] = tmp_column[0:height - 1 - shift]
                img[0:shift - 1, i] = np.flip(tmp_column[:shift - 1])
            if shift < 0:
                img[0:height - 1 + shift, i] = tmp_column[abs(shift):height - 1]
                img[height - 1 + shift:height - 1, i] = np.flip(tmp_column[height - 1 + shift:height - 1])

    elif len(img.shape) == 1:
        for i in range(len(shift_list)):
            shift = int(shift_list[i])
            img[i] = img[i] + shift
    return img


def medfilt(x, k):
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.median(y, axis=1)


def signaltonoise(a, axis, ddof):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m / sd)


# dark_to_bright, bright_to_dark = get_gradients(img,filter)
# filter: 'Scharr' or 'Sobel'
def get_gradients(img, filter):
    if filter == 'Scharr':
        img = cv2.Scharr(img, cv2.CV_16S, 0, 1)
        norm_Img = img * (1 / np.amax(img))
        return (norm_Img + abs(norm_Img)) / 2, (norm_Img - abs(norm_Img)) / (-2)

    if filter == 'Sobel':
        img = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=5)
        norm_Img = img * (1 / np.amax(img))
        return (norm_Img + abs(norm_Img)) / 2, (norm_Img - abs(norm_Img)) / (-2)


def find_nearest_return_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# y x
def path_to_y_array(paths, width):
    paths = np.array(sorted(np.array(paths), key=lambda tup: tup[0]))

    ys = paths[:, 0]
    reduced = [paths[np.argmax(paths[:, 1] == i)] for i in range(width)]
    reduced = np.array(sorted(np.array(reduced), key=lambda tup: tup[1]))
    reduced = reduced[:, 0]
    return reduced


def mask_image(img, y_values, offset=0, above=False):
    img = np.array(img.copy())
    height, width = img.shape

    for x in range(width):
        cur_col = img[:, x]
        for y in range(height):
            if above:
                if y < y_values[x] + offset:
                    cur_col[y] = 0.
            elif y > y_values[x] + offset:
                img[y, x] = 0.
    return img


################################ examples################################

def exampleFlatten():
    image_file = '../testvectors/sick.tif'
    im = io.imread(image_file, as_gray=True)
    im = im.astype('float32')
    flattened, shiftlist = flatten(im)
    plt.gca().imshow(flattened, cmap='gray')
    plt.setp(plt.gca(), xticks=[], yticks=[])
    plt.show()

# exampleIntensityProfiling()
