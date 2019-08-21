import itertools
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.filters import threshold_otsu

from assets import segmentation_helper


#
class Graph(object):
    def __init__(self, numNodes):
        # self.adjacencyMatrix = np.empty([numNodes, numNodes],dtype=int)
        self.adjacencyMatrix = dok_matrix((numNodes, numNodes), dtype=float)
        self.numNodes = numNodes

    def addEdge(self, start, end, value):
        # dict.__setitem__(self.adjacencyMatrix, (start,end), value)
        self.adjacencyMatrix[start, end] = value

    def removeEdge(self, start, end):
        if self.adjacencyMatrix[start, end] == 0:
            print("There is no edge between %d and %d" % (start, end))
        else:
            self.adjacencyMatrix[start, end] = 0

    def containsEdge(self, start, end):
        if self.adjacencyMatrix[start, end] > 0:
            return True
        else:
            return False

    def __len__(self):
        return self.numNodes


#
def find_path(img):
    height, width = img.shape

    # Defines a translation from 2 coordinates to a single number
    def to_index(y, x):
        return y * width + x

    # Defines a reversed translation from index to 2 coordinates
    def to_coordinates(index):
        return (int(index / width), int(index % width))

    # Defines weight function
    def weights(start, end):
        # 𝑾𝒊𝒋 = 𝟐 − (𝒈𝒊 + 𝒈𝒋) + 𝑾𝒎𝒊𝒏
        minWeight = 1E-5
        w = 2 - (start + end) + minWeight
        return w

    # Constructs path from predecessors
    def get_path(Pr, j):
        path = [j]
        k = j
        while Pr[k] != -9999:
            path.append(Pr[k])
            k = Pr[k]
        return path[::-1]

    start_time = time.time()
    # Create an instance of the graph with width*height nodes
    graph = Graph(height * width)

    directions = list(itertools.product([-1, 0, 1], [0, 1]))

    for i in range(0, height):
        for j in range(0, width):
            for i1, j1 in directions:
                if i == i + i1 and j == j + j1:
                    continue
                elif (i + i1 >= 0) and (i + i1 < height) and (j + j1 >= 0) and (j + j1 < width):
                    graph.addEdge(to_index(i, j), to_index(i + i1, j + j1),
                                  weights(img.item(i, j), img.item(i + i1, j + j1)))

                    if ((j == 0) or (j == width - 1)) and j1 == 0:
                        graph.addEdge(to_index(i, j), to_index(i + i1, j + j1), 1E-5)

    # Apply Dijkstra algorith for finding the shortest path
    dist_matrix, predecessors = dijkstra(csgraph=graph.adjacencyMatrix, indices=0, directed=False,
                                         return_predecessors=True)

    # list with all nodes between start and end point
    path = get_path(predecessors, len(dist_matrix) - 1)

    # log time
    elapsed_time = time.time() - start_time
    print("Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(
        int(elapsed_time % 60)) + " seconds")

    return [to_coordinates(p) for p in path if p % width != 0 and p % width != width - 1]


# dark_to_bright, bright_to_dark = get_gradients(img,filter)
# filter: 'Scharr' or 'Sobel'
def get_gradients(img, filter):
    if filter == 'Scharr':
        img = cv2.Scharr(img, cv2.CV_16S, 0, 1)
        norm_Img = (img - np.amin(img))/(np.amax(img)-np.amin(img))
        return norm_Img, (1-norm_Img)

    if filter == 'Sobel':
        img = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=5)
        norm_Img = img * (1 / np.amax(abs(img)))
        return (norm_Img + abs(norm_Img)) / 2, (norm_Img - abs(norm_Img)) / (-2)

    if filter == 'Sobel2':
        img = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
        norm_Img = (img - np.amin(img))/(np.amax(img)-np.amin(img))
        return norm_Img, (1-norm_Img)

def which_layer(img, path):


    # Gaussian
    gaussian = cv2.GaussianBlur(img, (5, 5), 0.8)

    # Otsu's thresholding
    thrsh = threshold_otsu(gaussian)
    norm_binary_img = (gaussian > thrsh).astype(int)

    # Initialization
    sum_pixel_values = 0
    number_pixel = 0

    # Iteration
    for i in range(1,len(path)-2):
        sum_pixel_values = sum_pixel_values + np.sum(norm_binary_img[0:path[i] - 1, i])
        number_pixel = number_pixel + path[i]

    print(sum_pixel_values / number_pixel)

    # Return condition
    if (sum_pixel_values / number_pixel) >= 0.025:
        return 'IS-OS'
    else:
        return 'Vitreous-NFL'


###############################PROGRAM###############################

#original = cv2.imread('../testvectors/sick/with druses/7F87A800.tif', 0)
path = 'C:/Users/ylliv/Documents/Medientechnologie/Medientechnologie/6.Semester/Retina/Python/Reduced_data_set/Reduced_data_set/sick/with_druses/'
name = '7CE97D80.tif'
original = cv2.imread(path + name, 0)

# pre-processing
resized = cv2.resize(original, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  # why resize?
crop, first_white, last_white = segmentation_helper.find_roi(resized)
flattened, invert = segmentation_helper.flatten(crop)
#padded = cv2.copyMakeBorder(flattened, 0, 0, 1, 1, cv2.BORDER_CONSTANT, value=0)
gblur = cv2.GaussianBlur(flattened, (5, 5), 3)

# shared values
gradient, gradient_negative = get_gradients(gblur, 'Scharr')
height, width = gradient.shape
img = flattened

# find first layer
first_layer = find_path(gradient)
first_layer = segmentation_helper.path_to_y_array(first_layer, width)
#print(which_layer(crop, y_list))

# find second layer
masked = segmentation_helper.mask_image(gradient, first_layer, offset=20, above=True)
second_layer = find_path(masked)
second_layer = segmentation_helper.path_to_y_array(second_layer, width)

#second layer bottom boundary
negmasked = segmentation_helper.mask_image(gradient_negative, first_layer, offset=20, above=True)
bottom_boundary = find_path(negmasked)
bottom_boundary = segmentation_helper.path_to_y_array(bottom_boundary, width)

#unflatten all
unflatten = segmentation_helper.shiftColumn(invert, img)
first_layer = segmentation_helper.shiftColumn(invert, np.array(first_layer))
second_layer = segmentation_helper.shiftColumn(invert, np.array(second_layer))
bottom_boundary = segmentation_helper.shiftColumn(invert, np.array(bottom_boundary))

#draw path
paths = np.zeros([img.shape[0], img.shape[1], 4], dtype=int)

for x in range(width):
    paths[first_layer[x], x] = (255, 0, 0, 255)
    paths[second_layer[x], x] = (0, 255, 0, 255)
    paths[bottom_boundary[x], x] = (0, 0, 255, 255)

plt.imshow(crop, cmap='gray', alpha=1)
plt.imshow(paths, cmap='gray')
plt.show()
