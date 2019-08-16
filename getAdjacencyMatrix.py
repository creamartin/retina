import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


#####################################


# Defines a translation from 2 coordinates to a single index number
def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    return ind

# Defines a reversed translation from index to 2 coordinates
def ind2sub(array_shape, ind):
    rows = (ind // array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

# constructs the adjacency matrices from an input image
def get_adjacency_matrix(img):

    # pad image with vertical column on both sides
    img = cv2.copyMakeBorder(img, 0, 0, 1, 1, cv2.BORDER_CONSTANT, value=0)

    # get  vertical gradient image
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    # norm values between 0 and 1
    gradImg = (gy - np.amin(gy))/(np.amax(gy)-np.amin(gy))

    # get the invert of the gradient image.
    gradImgMinus = 1-gradImg

    height, width = gradImg.shape
    
    # minimal weight
    minWeight = 1E-5

    # iteration order over neighboring pixels in X and Y direction
    neighborIterY = [1, 0, -1, 1, -1,  1,  0, -1]
    neighborIterX = [1, 1,  1, 0,  0, -1, -1, -1]


    # get location A (in the image as indices) for each weight.
    adjMAsub = np.arange(height*width)

    # convert adjMA to subscripts
    adjMAy,adjMAx = ind2sub(gradImg.shape, adjMAsub)

    adjMAsub = adjMAsub.reshape(adjMAsub.size,1)
    szadjMAsub = adjMAsub.shape

    #% prepare to obtain the 8-connected neighbors of adjMAsub
    # Construct an array by repeating neighborIterX the number of times given in the second argument
    neighborIterY = np.tile(neighborIterY, (szadjMAsub[0],1))
    neighborIterX = np.tile(neighborIterX, (szadjMAsub[0],1))
    
    # % repmat to [8,1]
    adjMAsub = np.tile(adjMAsub,(1, 8))
    adjMAy = np.tile(adjMAy,(1, 8))
    adjMAx = np.tile(adjMAx,(1, 8))


    #  get 8-connected neighbors of adjMAsub, adjMBx, adjMBy and adjMBsub
    adjMBy = adjMAy + neighborIterY.T.flatten()    
    adjMBx = adjMAx + neighborIterX.T.flatten()

    #  make sure all locations are within the image.
    keepInd1 = np.logical_and(adjMBy >= 0, adjMBy < height)    
    keepInd2 = np.logical_and(adjMBx >= 0, adjMBx < width)
    keepInd = np.logical_and(keepInd1,keepInd2)

    adjMAsub = adjMAsub.T.flatten()
    adjMAsub = adjMAsub.reshape(1,adjMAsub.size)
    adjMAsub = adjMAsub[keepInd]
    adjMBy = adjMBy[keepInd]    
    adjMBx = adjMBx[keepInd]

    adjMBsub = sub2ind(gradImg.shape,adjMBy[:],adjMBx[:])

    # calculate weight
    gradImg1 = gradImg.flatten()
    gradImgMinus1 = gradImgMinus.flatten()
    adjMW = 2 - gradImg1[adjMAsub[:]] - gradImg1[adjMBsub[:]] + minWeight
    adjMmW = 2 - gradImgMinus1[adjMAsub[:]] - gradImgMinus1[adjMBsub[:]] + minWeight

    # pad minWeight on both sides
    imgTmp = np.zeros(gradImg.shape)
    imgTmp[:,0] = 1
    imgTmp[:,-1] = 1


    ind1, ind2 = np.nonzero(imgTmp[:] == 1)   # find all pixels equal 1
    indices = sub2ind(imgTmp.shape, ind1, ind2)
    imageSideInd = np.isin(adjMBsub,indices) # Test whether each element of the first array is also present in a second array
    adjMW[imageSideInd] = minWeight
    adjMmW[imageSideInd] = minWeight

    adjMatrixW = []
    adjMatrixMW = []

    return adjMatrixW, adjMatrixMW, adjMAsub, adjMBsub, adjMW, adjMmW, img

# constructs a SiPy Compressed Sparse Row matrix 
def sparse_matrix (adjMW, adjMAsub, adjMBsub, img):
    adjMatrix = csr_matrix((adjMW, (adjMAsub,adjMBsub)),shape=(img.shape[0]*img.shape[1],img.shape[0]*img.shape[1]), dtype=np.float)
    adjMatrix.eliminate_zeros()
    return adjMatrix


# Apply Dijkstra algorith for finding the shortest path
def find_shortest_path(adjMatrixW):
    dist_matrix, predecessors = dijkstra(
        csgraph=adjMatrixW, indices=0, directed=True, return_predecessors=True, min_only=False)
    return dist_matrix, predecessors

# Constructs path from predecessors
def get_path(Pr,  j):
    path = np.array([j])
    k = j
    while Pr[k] != -9999:
        path = np.append(path, Pr[k])
        k = Pr[k]
    return path


def plot_layers(img, paths):
    width = img.shape[1]
    layers = np.zeros([img.shape[0], img.shape[1], 4], dtype=np.uint8)
    colors =  [[255, 0, 0, 255],  [0, 255, 0, 255], [0, 0, 255, 255], [0, 255, 255, 255], [255, 0, 255, 255], [255, 255, 0, 255], [255, 100, 0, 255]]
    color_index = 0

    for path in paths:
        for ind in path.path:
            if ind % width != 0 and ind % width != width-1:
            # set value in image to color_index to trace the path
                cooY, cooX = ind2sub(layers.shape, ind)
                layers[cooY, cooX] = colors[color_index]
        color_index += 1


    # Show layers
    plt.imshow(img, cmap='gray')
    plt.imshow(layers, cmap='gray', alpha=1)
    plt.show()


##############################################
################   EXAMPLE   #################
##############################################


# get path of an image.
# folderPath = 'C:/Users/ylliv/Documents/Medientechnologie/Medientechnologie/6.Semester/Retina/Python/'
# folderPath = 'C:/Users/ylliv/Documents/Medientechnologie/Medientechnologie/6.Semester/Retina/Python/Reduced_data_set/Reduced_data_set/sick/with_druses/'
# folderPath = 'C:/Users/ylliv/Documents/Medientechnologie/Medientechnologie/6.Semester/Retina/Python/Reduced_data_set/Reduced_data_set/healthy/normal/'
# name = '38B062A0.tif'
name = '437D6CD0.tif'

# #myimg = cv2.imread(folderPath + name, 0)
# resized = cv2.resize(img, dsize=None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)

# # blur image
# #gblur = cv2.GaussianBlur(padded,(5,5),3)
# #gblur = cv2.bilateralFilter(padded,5,25,25)
# gblur = cv2.medianBlur(resized, 5)

# # start log time
# start_time = time.time()
# # #####################################
# adjMatrixW, adjMatrixMW, adjMAsub, adjMBsub, adjMW, adjMmW, newImg = get_adjacency_matrix(gblur)
# adjMatrixW = sparse_matrix (adjMW[:], adjMAsub[:], adjMBsub[:], newImg)
# adjMatrixМmW = sparse_matrix (adjMmW[:], adjMAsub[:], adjMBsub[:], newImg)

# dist_matrix, predecessors = find_shortest_path(adjMatrixW)
# dist_matrix2, predecessors2 = find_shortest_path(adjMatrixМmW)

# # list with all nodes between start and end point
# path = get_path(predecessors, len(dist_matrix)-1)
# path2 = get_path(predecessors2, len(dist_matrix2)-1)

# # #############################################
# # #log time
# # elapsed_time = time.time() - start_time
# # print("Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(
# #     int(elapsed_time % 60)) + " seconds")
# # print(elapsed_time)

# # #paths = [path, path2]
# # plot_layers(newImg, [path, path2])
