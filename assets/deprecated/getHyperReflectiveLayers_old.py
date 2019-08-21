import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from assets.getAdjacencyMatrix import get_adjacency_matrix, sub2ind, ind2sub, plot_layers


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

def ind2sub1(array_shape, ind):
    #ind[ind < 0] = -1
    #ind[ind >= array_shape[0]*array_shape[1]] = -1
    rows = (ind / array_shape[1])
    cols = ind % array_shape[1]
    #rows = (ind.astype('int') / array_shape[1])
    #cols = (ind.astype('int') % array_shape[1])
    return (rows.astype(int), cols)

def get_path1(Pr,  j):
    path = np.array([j])
    k = j
    while Pr[k] != -9999:
        path = np.append(path, Pr[k])
        k = Pr[k]
    return path

def getHyperReflectiveLayers(inputImg, param):

    #initiate parameters
    if  param.shrink_scale:
        shrinkScale = param.shrink_scale
    else: shrinkScale = 0.2

    if  param.offsets:
        offsets = param.offsets
    else: offsets = np.arange(-20,21)

    #offsets = np.arange(-20,21)
    isPlot = 0

    #shrink the image.
    szImg = inputImg.size
    #procImg = imresize(inputImg,constants.shrinkScale,'bilinear')
    procImg = cv2.resize(inputImg, dsize=None, fx=shrinkScale, fy=shrinkScale,
                    interpolation=cv2.INTER_LINEAR)

    #create adjacency matrices
    adjMatrixW, adjMatrixMW, adjMX, adjMY, adjMW, adjMmW, newImg = get_adjacency_matrix(procImg)
    
    #create roi for getting shortestest path based on gradient-Y image.
    # get  vertical gradient image
    gy = cv2.Sobel(newImg, cv2.CV_64F, 0, 1, ksize=5)
    gy = (gy - np.amin(gy))/(np.amax(gy)-np.amin(gy))

    szImgNew = newImg.shape
    roiImg = np.zeros(szImgNew)
    roiImg[gy > np.mean(gy[:])] = 1 
    print(np.mean(gy[:]))

    # % find at least 2 layers
    # path{1} = 1;
    # count = 1;
    # while ~isempty(path) && count <= 2
    paths = np.empty(2, dtype=object)
    count = 0

    while  count < 2:

        #add columns of one at both ends of images
        roiImg[:,0] = 1
        roiImg[:,-1] = 1

        # cv2.imshow("roi", roiImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # include only region of interst in the adjacency matrix
        ind1, ind2 = np.nonzero(roiImg[:]==1)   # find all pixels equal 1
        indices = sub2ind(roiImg.shape, ind1, ind2)
        includeX = np.in1d(adjMX, indices)
        includeY = np.in1d(adjMY, indices)
        keepInd = np.logical_and(includeX, includeY)
        #compile adjacency matrix
        #adjMatrix = sparse_matrix(adjMmW[keepInd],adjMX[keepInd],adjMY[keepInd],newImg)
        adjMatrix = csr_matrix((adjMW[keepInd], (adjMX[keepInd],adjMY[keepInd])),shape=(newImg.shape[0]*newImg.shape[1],newImg.shape[0]*newImg.shape[1]), dtype=np.float)
        #adjMatrix = csr_matrix((adjMmW[:], (adjMY[:],adjMX[:])),shape=(newImg.shape[0]*newImg.shape[1],newImg.shape[0]*newImg.shape[1]), dtype=np.float)

        #adjMatrix.eliminate_zeros()
        #get layer going from dark to light        
        dist_matrix,predecessors  = dijkstra(
            csgraph=adjMatrix, indices=0, directed=False, return_predecessors=True, min_only=False)
    
        path = get_path1(predecessors, len(dist_matrix)-1)
        #plot_layers(gy, [path])
        # get rid of first few points and last few points
        pathX,pathY = ind2sub(newImg.shape,path)       
        

        #pathX = pathX([i] != [i+1])
        #grad = np.gradient(pathY) 
        pathX = pathX[np.gradient(pathY) != 0]
        pathY = pathY[np.gradient(pathY) != 0]

        #block the obtained path and abit around it
        pathXArr = np.tile(pathX, (len(offsets),len(offsets)))
        pathYArr = np.tile(pathY, (len(offsets),len(offsets)))

        for i in range(offsets.size):
            pathYArr[i,:] = pathYArr[i,:] + offsets[i]
        # pathXArr = repmat(pathX,numel(constants.offsets));
        # pathYArr = repmat(pathY,numel(constants.offsets));
        #     for i = 1:numel(constants.offsets)
        #         pathYArr(i,:) = pathYArr(i,:)+constants.offsets(i);
        #     end
        pathXArr = pathXArr[np.logical_and(pathYArr >= 0, pathYArr < szImgNew[1])]
        pathYArr = pathYArr[np.logical_and(pathYArr >= 0, pathYArr < szImgNew[1])]
        # pathXArr = pathXArr(pathYArr > 0 & pathYArr <= szImgNew(2));
        # pathYArr = pathYArr(pathYArr > 0 & pathYArr <= szImgNew(2));

        pathArr = sub2ind(szImgNew,pathXArr,pathYArr)
        # cv2.imshow("roi", roiImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        roiImg[pathXArr,pathYArr] = 0
        # cv2.imshow("roi", roiImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # pathArr = sub2ind(szImgNew,pathXArr,pathYArr);
        # roiImg(pathArr) = 0;

        indi = sub2ind(roiImg.shape, pathX, pathY)
        plot_layers(gy, [pathArr])

        paths[count] =  Path("", path, pathX, pathY)
        #paths = np.append(paths,Path("", path, pathX, pathY))


        count+=1

    #     %format paths back to original size
    # for i = 1:numel(paths)    
    #     [paths(i).path, paths(i).pathY, paths(i).pathX] = resizePath(szImg, szImgNew, constants, paths(i).pathY, paths(i).pathX);    
    #     paths(i).pathXmean = nanmean(paths(i).pathX);
    #     paths(i).name = [];
        
    # end

        #format paths back to original size
        #for i in range(paths.size):
        #path, pathY, pathX = resizePath(szImg, szImgNew, constants, pathY, pathX);    

    
    if paths[0].getPathXmean() > paths[1].getPathXmean():
        paths[0].name = 'isos'
        paths[1].name = 'ilm'
    else:
        paths[0].name = 'ilm'
        paths[1].name = 'isos'

    return paths

##############################################
################   Example   #################

# get path of an image.
folderPath = 'C:/Users/ylliv/Documents/Medientechnologie/Medientechnologie/6.Semester/Retina/Python/'
#folderPath = 'C:/Users/ylliv/Documents/Medientechnologie/Medientechnologie/6.Semester/Retina/Python/Reduced_data_set/Reduced_data_set/sick/with_druses/'
#folderPath = 'C:/Users/ylliv/Documents/Medientechnologie/Medientechnologie/6.Semester/Retina/Python/Reduced_data_set/Reduced_data_set/healthy/normal/'
#name = '38B062A0.tif'
#name = '437D6CD0.tif'
name = 'exampleOCTimage0001.tif'

myimg = cv2.imread(folderPath + name, 0)
# mypaths = getHyperReflectiveLayers(myimg)
