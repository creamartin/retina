# This function organize the workflow of the retinal segmentation
# It defines the boundaries between 7 retinal layers:

# ILM (vitreous-NFL)
# NFL/GCL
# IPL/INL
# INL/OPL
# OPL/ONL
# IS/OS 
# RPE

import cv2, json
import numpy as np

from getAdjacencyMatrix import get_adjacency_matrix, plot_layers
from getRetinalLayersCore import get_retinal_layers_core
from segmentation_helper import flatten


# This Object contains all the neccessary parameters of the segmentation 
class Params(object):
    def __init__(self):
        self.is_resize = 1.0
        self.filter_0_params = np.array([5, 5, 1])
        self.filter_params = np.array([20, 20, 2])
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

        # flatten adjacency matrices parameter
        self.adjMatrixW_f = []
        self.adjMatrixMW_f = []
        self.adjMAsub_f = []
        self.adjMBsub_f = []
        self.adjMW_f = []
        self.adjMmW_f = []

        # inner class rough_ilm_and_isos
        self.rough_ilm_and_isos = self.Rough_ilm_and_isos()

    class Rough_ilm_and_isos(object):
        def __init__(self):
            self.shrink_scale = 1.0
            self.offsets = np.arange(-5, 6)



def get_retinal_layers(img):


    # Parameter object:
    params = Params()


    # get Image size
    sz_img = img.shape


    # Pre-Processing ######################################
    # resize Image
    # img = cv2.resize(img, dsize=None, fx=1.0, fy=1.0, interpolation=cv2.INTER_AREA)

    # flatten image 
    img_f, unflatten = flatten(img)
    print(unflatten[1])

    # smooth image 
    # 2 approaches:

    ## Gaussian Blur
    # img = cv2.GaussianBlur(img,(3,3),0.8,0)

    ## Median
    img = cv2.medianBlur(img, 3)

    #######################################################

    # Create adjacency matrices 
    params.adjMatrixW, params.adjMatrixMW, params.adjMAsub, params.adjMBsub, params.adjMW, params.adjMmW, img_new = get_adjacency_matrix(
        img)

    params.adjMatrixW_f, params.adjMatrixMW_f, params.adjMAsub_f, params.adjMBsub_f, params.adjMW_f, params.adjMmW_f, img_new_f = get_adjacency_matrix(
        img_f)

        
    # Main part ###########################################
    # Iterate through a sorted list of layer names based on knowledge about the image struture  ###########################################
    
    # Pre-set order 
    retinal_layer_segmentation_order = ['roughILMandISOS', 'ilm', 'isos', 'rpe', 'inlopl', 'nflgcl', 'iplinl', 'oplonl']

    # Iterate through the list and save the found layer boundaries
    retinal_layers = []  
    for layer in retinal_layer_segmentation_order:
        retinal_layers = get_retinal_layers_core(layer, img_new, params, retinal_layers, unflatten)

    ########################################################

    return retinal_layers, img_new


def save_layers_to_file(layers, filename):
    layers_as_json = "{\"img_name\" : \"" + img_name + "\", \"layers\": ["
    for i, layer in enumerate(layers):
        layers_as_json += layer.JSON()
        if i < len(layers) - 1:
            layers_as_json += (",")
    layers_as_json += ("]}")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(json.loads(layers_as_json), f, ensure_ascii=False, indent=4)


##################### Example Code #########################
img_name = '7F87A800.tif'
save_dir = ""

img = cv2.imread(img_name, 0)
imglayers, img_new = get_retinal_layers(img)
save_layers_to_file(imglayers, str(img_name.split(".")[0]) + '.json')
plot_layers(img, imglayers)
