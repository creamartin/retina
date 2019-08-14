import numpy as np
import cv2
from getAdjacencyMatrix import get_adjacency_matrix, sparse_matrix, find_shortest_path,get_path, sub2ind,ind2sub, plot_layers
#import getRetinalLayers 

class Rough_ilm_and_isos(object):
    
    def __init__(self):
        #self.shrink_scale = 0.2
        self.shrink_scale = 1.0
        #self.offsets = np.arange(-20,21)
        self.offsets = np.arange(-5,6)


class Path(object):


    def __init__(self, name, path, pathX, pathY):
        self.name = name
        self.path = path
        self.pathX = pathX
        self.pathY = pathY
        self.pathYmean = np.mean(self.pathX)

    def getName(self):
        return self.name

    def setName(self, name):
        self.name = name

    def getPath(self):
        return self.path

    def setPath(self, path):
        self.path = path

    def getPathX(self):
        return self.pathX

    def setPathX(self, pathX):
        self.pathX = pathX

    def getPathY(self):
        return self.pathY

    def setPathY(self, pathY):
        self.pathY = pathY
        self.pathYmean = np.mean(self.pathY)

    def getPathYmean(self):
        return self.pathYmean





##############################################
################   EXAMPLE   #################
##############################################


# get path of an image.
# folderPath = 'C:/Users/ylliv/Documents/Medientechnologie/Medientechnologie/6.Semester/Retina/Python/'
# #folderPath = 'C:/Users/ylliv/Documents/Medientechnologie/Medientechnologie/6.Semester/Retina/Python/Reduced_data_set/Reduced_data_set/sick/with_druses/'
# #folderPath = 'C:/Users/ylliv/Documents/Medientechnologie/Medientechnologie/6.Semester/Retina/Python/Reduced_data_set/Reduced_data_set/healthy/normal/'
# #name = '38B062A0.tif'
# name = '437D6CD0.tif'
# #name = 'exampleOCTimage0001.tif'

# param = Rough_ilm_and_isos()
# myimg = cv2.imread(folderPath + name, 0)
# mypaths = getHyperReflectiveLayers(myimg,param)
