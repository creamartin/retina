import numpy as np
import cv2
import scipy as sc
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import dijkstra, shortest_path
from matplotlib import pyplot as plt
import itertools
import time
#from  skimage.filters import scharr
import find_roi as fr

#get path of an image.
folderPath = 'C:/Users/ylliv/Documents/Medientechnologie/Medientechnologie/6.Semester/Retina/Python/'
name = '38B062A0.tif'
img = cv2.imread(folderPath + name, 0)
resized = cv2.resize(img, dsize=None, fx=0.5, fy=0.5, interpolation = cv2.INTER_AREA)
#resized = cv2.resize(img, (300, 300), interpolation = cv2.INTER_AREA)

crop, first_white, last_white = fr.find_roi(resized)

padded= cv2.copyMakeBorder(crop,0,0,1,1,cv2.BORDER_CONSTANT,value=0)

#blur image
gblur = cv2.GaussianBlur(padded,(5,5),3)
# get  vertical gradient image

sobely = cv2.Sobel(gblur,cv2.CV_64F,0,1,ksize=5)
#norm values between 0 and 1
#sobely = scharr(resized)
gradImg = (sobely - np.amin(sobely))/(np.amax(sobely)-np.amin(sobely))
#cv2.imshow("pos", gradImg)

# get the "invert" of the gradient image.
gradImgMinus = 1-gradImg
# cv2.imshow("neg", gradImgMinus)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# pad image with vertical column on both sides
#img2= cv2.copyMakeBorder(gradImg,0,0,1,1,cv2.BORDER_CONSTANT,value=0)
#img3= cv2.copyMakeBorder(gradImgMinus,0,0,1,1,cv2.BORDER_CONSTANT,value=0)
img2 = gradImg
img3 = gradImgMinus

height,width = img2.shape

# Defines a translation from 2 coordinates to a single number
def to_index(y, x):
    return y * width + x

# Defines a reversed translation from index to 2 coordinates
def to_coordinates(index):
    return index / width, index % width

# Defines weight function
def weights(start, end):
	#𝑾𝒊𝒋 = 𝟐 − (𝒈𝒊 + 𝒈𝒋) + 𝑾𝒎𝒊𝒏
	minWeight = 1E-5
	w = 2 - (start+end) + minWeight
	return w

# Constructs path from predecessors 
def get_path(Pr,  j):
    path = [j]
    k = j
    while Pr[ k] != -9999:
        path.append(Pr[ k])
        k = Pr[ k]
    return path[::-1]


class Graph(object):
	def __init__(self, numNodes):
		#self.adjacencyMatrix = np.empty([numNodes, numNodes],dtype=int)
		self.adjacencyMatrix = dok_matrix((numNodes, numNodes),dtype=float)
		self.numNodes = numNodes

	def addEdge(self, start, end, value): 
        #dict.__setitem__(self.adjacencyMatrix, (start,end), value)
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

start_time = time.time()

# Create an instance of the graph with width*height nodes
graph1 = Graph(height*width)
graph2 = Graph(height*width)     

directions = list(itertools.product([-1, 0, 1], [-1, 0, 1]))

for i in range(0, height):
    for j in range(0, width):
        for i1, j1 in directions:
            if i == i+i1 and j == j+j1: continue
            elif (i+i1 >= 0)  and (i+i1 < height) and (j+j1 >= 0) and (j+j1 < width):
                 graph1.addEdge(to_index(i, j),to_index(i+i1,j+j1), weights(img2.item(i,j), img2.item(i+i1,j+j1)))
                 graph2.addEdge(to_index(i, j),to_index(i+i1,j+j1), weights(img3.item(i,j), img3.item(i+i1,j+j1)))

                 if ((j == 0) or (j == width-1)) and j1 == 0:
                     graph1.addEdge(to_index(i, j),to_index(i+i1,j+j1),1E-5)
                     graph2.addEdge(to_index(i, j),to_index(i+i1,j+j1),1E-5)

                    #  print(str(to_index(i, j)) + ": " + str(to_index(i+i1,j+j1)))

#print(graph1.adjacencyMatrix.toarray())    

#Appply Dijkstra algorith for finding the shortest path                
dist_matrix, predecessors = dijkstra(csgraph=graph1.adjacencyMatrix, indices=0, directed=False, return_predecessors=True, min_only=False)					
dist_matrix2, predecessors2 = dijkstra(csgraph=graph2.adjacencyMatrix, indices=0, directed=False, return_predecessors=True, min_only=False)					

# dist_matrix, predecessors = shortest_path(csgraph=graph1.adjacencyMatrix, method='BF',directed=False, indices=0, return_predecessors=True)
# dist_matrix2, predecessors2 = shortest_path(csgraph=graph2.adjacencyMatrix, method='BF',directed=False, indices=0, return_predecessors=True)

#print(dist_matrix[len(dist_matrix)-1])

# list with all nodes between start and end point
path = get_path(predecessors, len(dist_matrix)-1)
path2 = get_path(predecessors2, len(dist_matrix2)-1)

#print(path)

# log time
elapsed_time = time.time() - start_time
print("Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(
            int(elapsed_time % 60)) + " seconds")		
print(elapsed_time)	


layers = np.zeros([height, width, 4], dtype=np.uint8)

for ind in path:
	cooX,cooY = to_coordinates(ind)
	# set value in image to black to trace the path
	#padded1[int(cooX),cooY] = 255
	layers[int(cooX),cooY] = [255, 128, 0, 255]
for ind2 in path2:
	cooX,cooY = to_coordinates(ind2)
	# set value in image to black to trace the path
	#padded1[int(cooX),cooY] = 100
	layers[int(cooX),cooY] = [0, 128, 0, 255]



#cv2.imwrite(name + '_area.png',padded)

#plt.subplot(),plt.imshow(padded,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(img3,cmap = 'gray')
#plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
#plt.show()

im1 = plt.imshow(padded, cmap = 'gray')
im2 = plt.imshow(layers, cmap = 'gray', alpha=1)
plt.show()