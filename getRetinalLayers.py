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
from getAdjacencyMatrix import get_adjacency_matrix, sparse_matrix, find_shortest_path,get_path, sub2ind,ind2sub, plot_layers
from getRetinalLayersCore import get_retinal_layers_core
class Params:
    def __init__(self):
        self.is_resize = []
        self.filter_0_params = []
        self.ilm_0 = []
        self.ilm_1 = []
        self.is_os_0 = []
        self.is_os_1 = []
        self.rpe_0 = []
        self.rpe_1 = []
        self.inl_opl_0 = []
        self.inl_opl_1 = []
        self.nfl_gcl_0 = []
        self.nfl_gcl_1 = []
        self.ipl_inl_0 = []
        self.ipl_inl_1 = []
        self.opl_onl_0 = []
        self.opl_onl_1 = []

        # adjacency matrices parameter
        self.adj_matrix_w = []
        self.adj_matrix_m_w = []
        self.adj_m_a = []
        self.adj_m_b = []
        self.adj_M_w = []
        self.adj_M_m_w = []

        # inner class rough_ilm_and_isos
        self.rough_ilm_and_isos = self.Rough_ilm_and_isos()


    class Rough_ilm_and_isos:
        

        def __init__(self,):
            self.shrink_scale = []
            self.offsets = []






def get_retinal_layers(img):

    # get Image size
    sz_img = img.shape()

    # resize Image
    img = cv2.resize(img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # smooth image 
    # 2 approaches:

    ## Gaussian Blur
    #img = cv2.GaussianBlur(img,[3,3],1,0)

    ## Median
    img = cv2.medianBlur(img,[3,3])

    # Create adjacency matrices 
    adjMatrixW, adjMatrixMW, adjMAsub, adjMBsub, adjMW, adjMmW, img_new = get_adjacency_matrix(img)

    # Pre-set order 
    retinal_layer_segmentation_order = ['rough_ILM_and_ISOS', 'ilm', 'isos', 'rpe', 'inlopl', 'nflgcl', 'iplinl', 'oplonl']

    # Iterate through the list and save the found layer boundaries
    retinal_layers =[]
    for layer in  retinal_layer_segmentation_order:
        retinal_layers = get_retinal_layers_core(layer,img_new,params,retinal_layers)


    # test 










