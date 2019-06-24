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


# example
def example():
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

# example()
