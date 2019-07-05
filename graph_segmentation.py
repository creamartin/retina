import numpy as np
import cv2
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path
from matplotlib import pyplot as plt
import itertools
import time
import segmentation_helper


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

    return [to_coordinates(p) for p in path if p % width != 0 and p % width != width-1]


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


###############################PROGRAM###############################

img = cv2.imread('../testvectors/sick/with druses/7CE97D80.tif', 0)

# pre-processing
resized = cv2.resize(img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)  # why resize?
crop, first_white, last_white = segmentation_helper.find_roi(resized)
padded = cv2.copyMakeBorder(crop, 0, 0, 1, 1, cv2.BORDER_CONSTANT, value=0)
gblur = cv2.GaussianBlur(padded, (5, 5), 3)

# find first layer
gradient, gradient_negative = get_gradients(gblur, 'Scharr')

first_layer = find_path(gradient)
print(crop.shape)
#result =
for p in first_layer:
    y = p[0]
    x = p[1]
    print(p)
    crop[y-1,(x-1)] = 1.
    #cv2.

plt.imshow(crop, cmap='gray',alpha=1)
plt.show()
