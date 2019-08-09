import numpy as np
import cv2
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path
from matplotlib import pyplot as plt
import itertools
import time
import segmentation_helper

# This function is used in get_retinal_layers

def get_retinal_layers_core(layer_name, img, params,paths_list):


    if layer_name ==  'roughILMandISOS':

        img_old    = img[:,1:img.shape[1]-2]
        temp_paths = get_hyper_reflective_layers(imgOld,params.rough_ILM_and_ISOS) # Get the rough segmentation of the first prominate layers ILM and ISOS
        paths_list = None
        
        return temp_paths


    elif layer_name in ['nflgcl' 'inlopl' 'ilm' 'isos' 'oplonl' 'iplinl' 'rpe']:
        
        adj_ma  = params.adj_m_a
        adj_mb  = params.adj_m_b
        adj_MW  = params.adj_M_w
        adj_MmW = params.adj_M_m_w

    
    # init region of interest
    sz_img  = img.shape()
    roi_img = np.zeros([sz_img])

    for k in range(1, sz_img[2] -2):

        if layer_name == 'nflgcl':

            # region between 'ilm and 'inlopl'

            ind_pathX   = np.where(next((x for x in paths_list if x.name == 'ilm'), None).pathY == k)
            start_ind_0 = next((x for x in paths_list if x.name == 'ilm'), None).pathX(ind_pathX(0))
            ind_pathX   = np.where(next((x for x in paths_list if x.name == 'inlopl'), None).pathY == k)
            end_ind_0 = next((x for x in paths_list if x.name == 'inlopl'), None).pathX(ind_pathX(0)) 

            start_ind = start_ind_0 # The matlab-code use an additional offset (params.nflgcl_0/1) -> able to add later on 
            end_ind   = end_ind_0

        elif layer_name == 'rpe':

            # region below the isos

            ind_pathX   = np.where(next((x for x in paths_list if x.name == 'ilm'), None).pathY == k)  
            start_ind_0 = next((x for x in paths_list if x.name == 'ilm'), None).pathX(ind_pathX(0))
            end_ind_0   = start_ind_0 + round(next((x for x in paths_list if x.name == 'isos'), None).pathXmean - next((x for x in paths_list if x.name == 'ilm'), None).pathXmean)      

            start_ind = start_ind_0 # The matlab-code use an additional offset (params.rpe_0/1) -> able to add later on 
            end_ind   = end_ind_0

        elif layer_name == 'inlopl':

            # region between 'ilm' and 'isos'

            ind_pathX   = np.where(next((x for x in paths_list if x.name == 'ilm'), None).pathY == k) 
            start_ind_0 = next((x for x in paths_list if x.name == 'ilm'), None).pathX(ind_pathX(0))

            ind_pathX = np.where(next((x for x in paths_list if x.name == 'isos'), None).pathY == k) 
            end_ind_0 = next((x for x in paths_list if x.name == 'isos'), None).pathX(ind_pathX(0))

            start_ind = start_ind_0 # The matlab-code use an additional offset (params.inlopl_0/1)  -> able to add later on 
            end_ind   = end_ind_0

        elif layer_name == 'ilm':

            # region near the previous rough segmented ilm path

            ind_pathX   = np.where(next((x for x in paths_list if x.name == 'ilm'), None).pathY == k) 

            start_ind = next((x for x in paths_list if x.name == 'ilm'), None).pathX(ind_pathX(0)) # - offset params.ilm_0
            end_ind   = next((x for x in paths_list if x.name == 'ilm'), None).pathX(ind_pathX(0)) # + offset params.ilm_1

        elif layer_name == 'isos':

            # region near the previous rough segmented isos path

            ind_pathX   = np.where(next((x for x in paths_list if x.name == 'isos'), None).pathY == k) 

            start_ind = next((x for x in paths_list if x.name == 'isos'), None).pathX(ind_pathX(0)) # - offset params.isos_0
            end_ind   = next((x for x in paths_list if x.name == 'isos'), None).pathX(ind_pathX(0)) # + offset params.isos_1

        elif layer_name == 'iplinl':

            # region between 'nflgcl' and 'inlopl'

            ind_pathX   = np.where(next((x for x in paths_list if x.name == 'nlfgcl'), None).pathY == k) 
            start_ind_0 = next((x for x in paths_list if x.name == 'nflgcl'), None).pathX(ind_pathX(0))

            ind_pathX = np.where(next((x for x in paths_list if x.name == 'inlopl'), None).pathY == k) 
            end_ind_0 = next((x for x in paths_list if x.name == 'inlopl'), None).pathX(ind_pathX(0)) 

            start_ind = start_ind_0 # The matlab-code use an additional offset (params.iplinl_0/1) -> able to add later on 
            end_ind   = end_ind_0

        elif layer_name == 'oplonl':

            # region between 'inlopl' and 'isos'

            ind_pathX   = np.where(next((x for x in paths_list if x.name == 'inlopl'), None).pathY == k) 
            start_ind_0 = next((x for x in paths_list if x.name == 'inlopl'), None).pathX(ind_pathX(0))

            ind_pathX = np.where(next((x for x in paths_list if x.name == 'isos'), None).pathY == k) 
            end_ind_0 = next((x for x in paths_list if x.name == 'isos'), None).pathX(ind_pathX(0))

            start_ind = start_ind_0 # The matlab-code use an additional offset (params.oplonl_0/1)  -> able to add later on 
            end_ind   = end_ind_0


        
        # error checking

        if start_ind > end_ind:
            start_ind = end_ind - 1

        if start_ind < 0:
            start_ind = 0

        if end_ind > sz_img(0) - 1:
            end_ind = sz_img(0) -1 


        # set column_wise region of interest to 1 
        roi_img[start_ind : end_ind,k] = 1
        roi_img[:,0] = 1
        roi_img[:,sz_img(1)-1] = 1

        # include only the region of interest in the adjacency matrix
        include_a = np.in1d(adj_ma,np.where(roi_img == 1))
        include_b = np.in1d(adj_mb,np.where(roi_img == 1))
        
        keep_ind = np.logical_and(include_a,include_b)


        # Black to white or white to black adjacency

        if ['rpe' 'nflgcl' 'oplonl' 'iplinl' ]:


        elif: ['inlopl' 'ilm' 'isos' ]:

            















    