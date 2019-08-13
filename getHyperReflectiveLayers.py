import numpy as np
import cv2
from getAdjacencyMatrix import get_adjacency_matrix, sparse_matrix, find_shortest_path,get_path, sub2ind,ind2sub, plot_layers
#import getRetinalLayers 

class Rough_ilm_and_isos(object):
    
    def __init__(self):
        #self.shrink_scale = 0.2
        self.shrink_scale = 1.0
        #self.offsets = np.arange(-20,21)
        self.offsets = np.arange(-10,11)


class Path(object):


    def __init__(self, name, path, pathX, pathY):
        self.name = name
        self.path = path
        self.pathX = pathX
        self.pathY = pathY
        self.pathXmean = np.mean(self.pathX)

    def getName(self):
        return self.name

    def setName(self, name):
        self.name = name

    def getPath(self):
        return self.path

    def setPath(self, path):
        self.path = path
        self.pathXmean = np.mean(self.path)

    def getPathX(self):
        return self.pathX

    def setPathX(self, pathX):
        self.pathX = pathX

    def getPathY(self):
        return self.pathY

    def setPathY(self, pathY):
        self.pathY = pathY

    def getPathXmean(self):
        return self.pathXmean


def getHyperReflectiveLayers(inputImg, param):

    #initiate parameters
    if param.shrink_scale is None:
        shrinkScale = 0.2
    else:
        shrinkScale = param.shrink_scale

    if param.offsets is None:
        offsets = np.arange(-20,21)
    else:
        offsets = param.offsets

    #resize the image.
    szImg = inputImg.size
    resizedImg = cv2.resize(inputImg, dsize=None, fx=shrinkScale, fy=shrinkScale,
                    interpolation=cv2.INTER_LINEAR)

    #create adjacency matrices
    adjMatrixW, adjMatrixMW, adjMX, adjMY, adjMW, adjMmW, newImg = get_adjacency_matrix(resizedImg)
    
    #create roi for getting shortestest path based on vertical gradient image.
    # get  vertical gradient image
    gy = cv2.Sobel(newImg, cv2.CV_64F, 0, 1, ksize=5)
    # normalize gradient
    gy = (gy - np.amin(gy))/(np.amax(gy)-np.amin(gy))

    # create binary mask
    szImgNew = newImg.shape
    roiImg = np.zeros(szImgNew)
    # set value 1 where pixel value is greater than the mean value of the gradient
    roiImg[gy > np.mean(gy[:])] = 1 

    # find 2 layers 
    paths = np.empty(2, dtype=object)
    count = 0

    while  count < len(paths):

        #add columns of value 1 at both ends of the image
        roiImg[:,0] = 1
        roiImg[:,-1] = 1

        # include only region of interst in the adjacency matrix
        ind1, ind2 = np.nonzero(roiImg[:] == 1)   # find all pixels equal 1
        indices = sub2ind(roiImg.shape, ind1, ind2)
        includeX = np.isin(adjMX, indices)  #Test whether each element of first array is also present in a second array
        includeY = np.isin(adjMY, indices)
        keepInd = np.logical_and(includeX, includeY)

        #compile adjacency matrix
        adjMatrix = sparse_matrix(adjMW[keepInd],adjMX[keepInd],adjMY[keepInd],newImg)
        
        # apply Dijkstra algorithm
        dist_matrix,predecessors = find_shortest_path(adjMatrix)

        # construct path from the predecessor nodes retrieved from Dijkstra algorithm 
        path = get_path(predecessors, len(dist_matrix)-1)

        # get rid of first few points and last few points
        pathX,pathY = ind2sub(newImg.shape,path)        
        pathX = pathX[np.gradient(pathY) != 0]
        pathY = pathY[np.gradient(pathY) != 0]

        #block the obtained path and abit around it
        pathXArr = np.tile(pathX, (len(offsets), len(offsets)))
        pathYArr = np.tile(pathY, (len(offsets), len(offsets)))

        for i in range(offsets.size):
            #pathYArr[i,:] = pathYArr[i,:] + offsets[i]
            pathXArr[i,:] = pathXArr[i,:] + offsets[i]

        
        
        #pathXArr = pathXArr[np.logical_and(pathYArr >= 0, pathYArr < szImgNew[1])]
        pathXArr = pathXArr[np.logical_and(pathXArr >= 0, pathXArr < szImgNew[0])]
        pathYArr = pathYArr[np.logical_and(pathYArr >= 0, pathYArr < szImgNew[1])]
        
        pathArr = sub2ind(szImgNew,pathXArr,pathYArr)
        
        roiImg[pathXArr,pathYArr] = 0
    
        # plot the masked path
        plot_layers(gy, [pathArr])

        paths[count] =  Path("", path, pathX, pathY)

        count += 1

    #format paths back to original size
   
    
    if paths[0].getPathXmean() > paths[1].getPathXmean():
        paths[0].name = 'isos'
        paths[1].name = 'ilm'
    else:
        paths[0].name = 'ilm'
        paths[1].name = 'isos'

    return paths

##############################################
################   EXAMPLE   #################
##############################################


# get path of an image.
folderPath = 'C:/Users/ylliv/Documents/Medientechnologie/Medientechnologie/6.Semester/Retina/Python/'
#folderPath = 'C:/Users/ylliv/Documents/Medientechnologie/Medientechnologie/6.Semester/Retina/Python/Reduced_data_set/Reduced_data_set/sick/with_druses/'
#folderPath = 'C:/Users/ylliv/Documents/Medientechnologie/Medientechnologie/6.Semester/Retina/Python/Reduced_data_set/Reduced_data_set/healthy/normal/'
#name = '38B062A0.tif'
name = '437D6CD0.tif'
#name = 'exampleOCTimage0001.tif'

param = Rough_ilm_and_isos()
myimg = cv2.imread(folderPath + name, 0)
mypaths = getHyperReflectiveLayers(myimg,param)
