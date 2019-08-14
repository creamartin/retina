import numpy as np
import cv2
import scipy as sc
from scipy.sparse import dok_matrix
from scipy.sparse.csgraph import dijkstra
from matplotlib import pyplot as plt
import itertools
import time
from  skimage.filters import scharr

#get path of an image.
folderPath = 'C:/Users/ylliv/Documents/Medientechnologie/Medientechnologie/6.Semester/Retina/Python/'
img = cv2.imread(folderPath + '437D6CD0.tif',0)
resized = cv2.resize(img,(300,300))

#blur image
gblur = cv2.GaussianBlur(resized,(5,5),3)
# get  vertical gradient image
sobely = cv2.Sobel(gblur,cv2.CV_64F,0,1,ksize=5)
#norm values between 0 and 1
#sobely = scharr(resized)
gradImg = (sobely - np.amin(sobely))/(np.amax(sobely)-np.amin(sobely))
# pad image with vertical column on both sides
img2= cv2.copyMakeBorder(gradImg,0,0,1,1,cv2.BORDER_CONSTANT,value=0)

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
directions = list(itertools.product([-1, 0, 1], [-1, 0, 1]))

for i in range(0, height):
    for j in range(0, width):
        for i1, j1 in directions:
            if i == i+i1 and j == j+j1: continue
            elif (i+i1 >= 0)  and (i+i1 < height) and (j+j1 >= 0) and (j+j1 < width):
                 graph1.addEdge(to_index(i, j),to_index(i+i1,j+j1), weights(img2[i,j], img2[i+i1,j+j1]))
                 if (j == 0) or (j == width-1):
                     graph1.addEdge(to_index(i, j),to_index(i+i1,j+j1),1E-5)
                    #  print(str(to_index(i, j)) + ": " + str(to_index(i+i1,j+j1)))

#print(graph1.adjacencyMatrix.toarray())    

#Appply Dijkstra algorith for finding the shortest path                
dist_matrix, predecessors = dijkstra(csgraph=graph1.adjacencyMatrix, indices=0, directed=False, return_predecessors=True, min_only=False)					
#print(dist_matrix[len(dist_matrix)-1])

# list with all nodes between start and end point
path = get_path(predecessors, len(dist_matrix)-1)
#print(path)

for ind in path:
	cooX,cooY = to_coordinates(ind)
	# set value in image to black to trace the path
	img2[int(cooX),cooY,] = 0
#print(predecessors)

#cv2.imshow("img2", img2)
#cv2.waitKey(0)

# log time
elapsed_time = time.time() - start_time
print("Execution time: " + str(int(elapsed_time / 60)) + " minute(s) and " + str(
            int(elapsed_time % 60)) + " seconds")		
print(elapsed_time)	

# plt.subplot(121),plt.imshow(resized,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(),plt.imshow(img2,cmap = 'gray')
#plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.show()
