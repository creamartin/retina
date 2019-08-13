# This function organize the workflow of the retinal segmentation
# It defines the boundaries between 7 retinal layers:

# ILM (vitreous-NFL)
# NFL/GCL
# IPL/INL
# INL/OPL
# OPL/ONL
# IS/OS
# RPE

import numpy as np
import cv2
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path
from matplotlib import pyplot as plt
import itertools
import time
import segmentation_helper
from getAdjacencyMatrix import get_adjacency_matrix, sparse_matrix, find_shortest_path, get_path, sub2ind,ind2sub, plot_layers
from getRetinalLayersCore import get_retinal_layers_core
import skimage.io as io
from segmentation_helper import flatten

class Params(object):

    def __init__(self):
        self.is_resize = 1.0
        self.filter_0_params = np.array([5,5,1])
        self.filter_params = np.array([20,20,2])
        self.ilm_0 = 2
        self.ilm_1 = 2
        self.is_os_0 = 4
        self.is_os_1 = 4
        self.rpe_0 = 0.05
        self.rpe_1 = 0.05
        self.inl_opl_0 = 0.1 
        self.inl_opl_1 = 0.3
        self.nfl_gcl_0 = 0.05
        self.nfl_gcl_1 = 0.3
        self.ipl_inl_0 = 0.6
        self.ipl_inl_1 = 0.2
        self.opl_onl_0 = 0.05
        self.opl_onl_1 = 0.5

        # adjacency matrices parameter
        self.adjMatrixW = []
        self.adjMatrixMW = []
        self.adjMAsub = []
        self.adjMBsub = []
        self.adjMW = []
        self.adjMmW = []

        # inner class rough_ilm_and_isos
        self.rough_ilm_and_isos = self.Rough_ilm_and_isos()


    class Rough_ilm_and_isos(object):
        

        def __init__(self):
            self.shrink_scale = 1.0
            self.offsets = np.arange(-10,11)



params = Params()



def get_retinal_layers(img):

    # get Image size
    sz_img = img.shape

    # resize Image
    img = cv2.resize(img, dsize=None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)

    # smooth image 
    # 2 approaches:

    ## Gaussian Blur
    #img = cv2.GaussianBlur(img,[3,3],1,0)

    ## Median
    img = cv2.medianBlur(img,3)

    # Create adjacency matrices 
    params.adjMatrixW, params.adjMatrixMW, params.adjMAsub, params.adjMBsub, params.adjMW, params.adjMmW, img_new = get_adjacency_matrix(img)

    # Pre-set order 
    retinal_layer_segmentation_order = ['roughILMandISOS', 'ilm', 'isos', 'rpe', 'inlopl', 'nflgcl', 'iplinl', 'oplonl']

    # Iterate through the list and save the found layer boundaries
    retinal_layers = []
    for layer in  retinal_layer_segmentation_order:
        retinal_layers = get_retinal_layers_core(layer,img_new,params,retinal_layers)


    return retinal_layers



##################### Example Code #########################

img = cv2.imread('61D70FB0.tif', 0)
imglayers = get_retinal_layers(img)











