import numpy as np
import cv2
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path
from matplotlib import pyplot as plt
import itertools
import time
import segmentation_helper
from getAdjacencyMatrix import get_adjacency_matrix , sparse_matrix, find_shortest_path,get_path, sub2ind,ind2sub, plot_layers
import getHyperReflectiveLayers as hrl

# path class
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



# This function is used in get_retinal_layers
def get_retinal_layers_core(layer_name, img, params,paths_list):


    if layer_name ==  'roughILMandISOS':

        img_old    = img[:,1:img.shape[1]-2]
        temp_paths = hrl.getHyperReflectiveLayers(img_old,params.rough_ilm_and_isos) # Get the rough segmentation of the first prominate layers ILM and ISOS
        paths_list = None
        
        return temp_paths


    elif layer_name in ['nflgcl', 'inlopl' ,'ilm' ,'isos', 'oplonl', 'iplinl' ,'rpe']:
        
        adj_ma  = params.adjMAsub
        adj_mb  = params.adjMBsub
        adj_MW  = params.adjMW 
        adj_MmW = params.adjMmW 


    # falla

    # init region of interest
    sz_img  = img.shape
    roi_img = np.zeros(sz_img)

    for k in range(0,img.shape[1]-1):

        if layer_name == 'nflgcl':

            # region between 'ilm and 'inlopl'

            ind_pathX   = np.where(next((x for x in paths_list if x.name == 'ilm'), None).pathY == k)
            ind_pathX   = ind_pathX[0].reshape(1,ind_pathX[0].size)
            start_ind_0 = next((x for x in paths_list if x.name == 'ilm'), None).pathX[ind_pathX[0,0]]

            ind_pathX   = np.where(next((x for x in paths_list if x.name == 'inlopl'), None).pathY == k)
            ind_pathX   = ind_pathX[0].reshape(1,ind_pathX[0].size)
            end_ind_0   = next((x for x in paths_list if x.name == 'inlopl'), None).pathX[ind_pathX[0,0]]

            start_ind = start_ind_0 +5# The matlab-code use an additional offset (params.nflgcl_0/1) -> able to add later on 
            end_ind   = end_ind_0 -5 

        elif layer_name == 'rpe':

            # region below the isos

            # use flattened image 

            ind_pathX   = np.where(next((x for x in paths_list if x.name == 'isos'), None).pathY == k)
            ind_pathX   = ind_pathX[0].reshape(1,ind_pathX[0].size) 

            start_ind_0 = next((x for x in paths_list if x.name == 'isos'), None).pathX[ind_pathX[0,0]]
            end_ind_0   = start_ind_0 + int(round(next((x for x in paths_list if x.name == 'isos'), None).pathXmean - next((x for x in paths_list if x.name == 'ilm'), None).pathXmean))      

            start_ind = start_ind_0 # The matlab-code use an additional offset (params.rpe_0/1) -> able to add later on 
            end_ind   = end_ind_0

        elif layer_name == 'inlopl':

            # region between 'ilm' and 'isos'

            ind_pathX   = np.where(next((x for x in paths_list if x.name == 'ilm'), None).pathY == k)
            ind_pathX = ind_pathX[0].reshape(1,ind_pathX[0].size)
            start_ind_0 = next((x for x in paths_list if x.name == 'ilm'), None).pathX[ind_pathX[0,0]]

            ind_pathX = np.where(next((x for x in paths_list if x.name == 'isos'), None).pathY == k)
            ind_pathX = ind_pathX[0].reshape(1,ind_pathX[0].size)
            end_ind_0 = next((x for x in paths_list if x.name == 'isos'), None).pathX[ind_pathX[0,0]]

            start_ind = start_ind_0 +10 # The matlab-code use an additional offset (params.inlopl_0/1)  -> able to add later on 
            end_ind   = end_ind_0 - 10

        elif layer_name == 'ilm':

            # region near the previous rough segmented ilm path

            ind_pathX = np.where(next((x for x in paths_list if x.name == 'ilm'), None).pathY == k) 
            ind_pathX = ind_pathX[0].reshape(1,ind_pathX[0].size)

            start_ind = next((x for x in paths_list if x.name == 'ilm'), None).pathX[ind_pathX[0,0]]  - params.ilm_0
            end_ind   = next((x for x in paths_list if x.name == 'ilm'), None).pathX[ind_pathX[0,0]]  + params.ilm_1

        elif layer_name == 'isos':

            # region near the previous rough segmented isos path

            # use flatten image 

            ind_pathX = np.where(next((x for x in paths_list if x.name == 'isos'), None).pathY == k) 
            ind_pathX = np.reshape(ind_pathX,(1,ind_pathX[0].size))

            start_ind = next((x for x in paths_list if x.name == 'isos'), None).pathX[ind_pathX[0,0]] - params.is_os_0
            end_ind   = next((x for x in paths_list if x.name == 'isos'), None).pathX[ind_pathX[0,0]] + params.is_os_1

        elif layer_name == 'iplinl':

            # region between 'nflgcl' and 'inlopl'

            ind_pathX   = np.where(next((x for x in paths_list if x.name == 'nflgcl'), None).pathY == k)
            ind_pathX = ind_pathX[0].reshape(1,ind_pathX[0].size) 
            start_ind_0 = next((x for x in paths_list if x.name == 'nflgcl'), None).pathX[ind_pathX[0,0]]

            ind_pathX = np.where(next((x for x in paths_list if x.name == 'inlopl'), None).pathY == k)
            ind_pathX = ind_pathX[0].reshape(1,ind_pathX[0].size) 
            end_ind_0 = next((x for x in paths_list if x.name == 'inlopl'), None).pathX[ind_pathX[0,0]]

            start_ind = start_ind_0 + 2# The matlab-code use an additional offset (params.iplinl_0/1) -> able to add later on 
            end_ind   = end_ind_0 - 3

        elif layer_name == 'oplonl':

            # region between 'inlopl' and 'isos'

            ind_pathX   = np.where(next((x for x in paths_list if x.name == 'inlopl'), None).pathY == k)
            ind_pathX = ind_pathX[0].reshape(1,ind_pathX[0].size) 
            start_ind_0 = next((x for x in paths_list if x.name == 'inlopl'), None).pathX[ind_pathX[0,0]]

            ind_pathX = np.where(next((x for x in paths_list if x.name == 'isos'), None).pathY == k)
            ind_pathX = ind_pathX[0].reshape(1,ind_pathX[0].size)
            end_ind_0 = next((x for x in paths_list if x.name == 'isos'), None).pathX[ind_pathX[0,0]]

            start_ind = start_ind_0 +3 # The matlab-code use an additional offset (params.oplonl_0/1)  -> able to add later on 
            end_ind   = end_ind_0 -5


        
        # error checking

        if start_ind > end_ind:
            start_ind = end_ind - 1

        if start_ind < 0:
            start_ind = 0

        if end_ind > sz_img[0] - 1:
            end_ind = sz_img(0) -1 
        


        # set column_wise region of interest to 1 
        roi_img[start_ind : end_ind,k] = 1


    # set first and last column to 1    
    roi_img[:,0] = 1
    roi_img[:,-1] = 1
       



    # include only the region of interest in the adjacency matrix
    ind1, ind2 = np.nonzero(roi_img[:] == 1)
    include_a = np.isin(adj_ma,sub2ind(roi_img.shape,ind1,ind2))
    include_b = np.isin(adj_mb,sub2ind(roi_img.shape,ind1,ind2))
        
    keep_ind = np.logical_and(include_a,include_b)           


    # Black to white or white to black adjacency

    # dark to white
    if layer_name in ['inlopl','ilm' ,'isos']:

        adjMatrixW = sparse_matrix(adj_MW[keep_ind],adj_ma[keep_ind],adj_mb[keep_ind],img)
        dist_matrix , predecessors = find_shortest_path(adjMatrixW)
        path = get_path(predecessors, len(dist_matrix) -1)
    
    # white to dark 
    elif layer_name in ['rpe', 'nflgcl' ,'oplonl' ,'iplinl' ]:

        adjMatrixMW = sparse_matrix(adj_MmW[keep_ind],adj_ma[keep_ind],adj_mb[keep_ind],img)
        dist_matrix , predecessors = find_shortest_path(adjMatrixMW)
        path = get_path(predecessors, len(dist_matrix) -1)

    # get pathX and pathY
    pathX, pathY = ind2sub(sz_img,path)






    # check if a layer is to overwrite like 'ilm' or 'isos' or if new layer is to added to paths_list
    matched_layers_id = True
    index = 0
    for layer in range(0,paths_list.size):
        if paths_list[layer].name == layer_name:
            matched_layers_id = False
            index = layer
    

    # save data
    if matched_layers_id:
        paths_list = np.append(paths_list,Path(layer_name, path, pathX, pathY))
    else:
        #test 
        paths_list[index] = None
        paths_list[index] = Path(layer_name, path, pathX, pathY)

    
         


    return paths_list       
            















    