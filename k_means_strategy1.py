from Precode import *
import numpy
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.io import loadmat

# use either one for the data file
# data = loadmat('allSamples_strategy1.mat')
# data = np.load('AllSamples.npy')

# three random coordinates for the first set of centroids
k1 = 3
i_point1 = [[ 7.78551305  3.12724529]
 [ 3.72610844  5.20432439]
 [ 3.2492998   5.59125171]]

# five random coordinates for the second set of centroids
k2 = 5
i_point2[[ 2.3537231   6.29810755]
 [ 2.81629029  3.1999725 ]
 [ 6.6161895   0.66750633]
 [ 5.38398051  3.53840433]
 [ 7.59731342  1.16504743]]

 # Assign the initial random centroids from the points
class KMeans:
    def __init__(self):
        self.coordinate
        self.centroids
        self.centroid_idx
        self.cost
        
    def setCoordinate(self, coordinate):
        self.coordinate = coordinate
        
    def setCentroids(self, centroids):
        self.centroids = centroids
        
    # Make point objects and assign them their coordinate, the closest centroid, and the cost 
    def closestCentroid(self):
        min_centroid_idx_list = []
        min_cost_list = []
        
        for c in range(len(self.coordinate)):
            min_centroid_idx = 0
            min_cost = float('inf')
            
            for i in range(len(self.centroids)):
                x = self.coordinate[c][0] - self.centroids[i][0]
                y = self.coordinate[c][1] - self.centroids[i][1]
                euclidean_norm = x**2 + y **2
                if min_cost > euclidean_norm:
                    min_centroid_idx = i 
                    min_cost = euclidean_norm
                    
            min_centroid_idx_list.append(min_centroid_idx)
            min_cost_list.append(min_cost)

        self.centroid_idx = min_centroid_idx_list
        self.cost = min_cost_list
    
    # Find the mean centroid for each cluster
    def clusterMeanCentroid(self):
        # counts the number of point to each centroid in popular_centroid
        popular_centroids = [0 for i in range(len(self.centroids))]
        # sum of the points close to each centroid
        centroids = [[0,0] for i in range(len(self.centroids))]
        
        for i in range(len(self.coordinate)):
            centroids[self.centroid_idx[i]][0] += self.coordinate[i][0]
            centroids[self.centroid_idx[i]][1] += self.coordinate[i][1]
            popular_centroids[self.centroid_idx[i]] += 1

        # divide by len of coordinate to get mean
        for i in range(len(centroids)):
            if popular_centroids[i] > 0:
                centroids[i][0] = centroids[i][0] / popular_centroids[i]
                centroids[i][1] = centroids[i][1] / popular_centroids[i]
            
        self.centroids = centroids
    
    # Repeat until the new mean centroid is the same as the last centroid
    def repeatKMeansClustering(self):
        # pre centroids
        self.closestCentroid(self)
        self.clusterMeanCentroid(self)
        pre_centroids = self.centroids
        num_repeat = 1
        
        # new centroids
        self.closestCentroid(self)
        self.clusterMeanCentroid(self)
        new_centroids = self.centroids
        num_repeat = 2
        
        while pre_centroids != new_centroids:
            num_repeat += 1
#             print("repeated #: ", num_repeat)
#             print("pre centroid: ", pre_centroids, "\nnew_centroids: ", new_centroids)
            pre_centroids = new_centroids
            self.closestCentroid(self)
            self.clusterMeanCentroid(self)
            new_centroids = self.centroids
        
        final_cost = sum(self.cost)

        return [self.centroids, final_cost]

# Set 1
point1 = KMeans
point1.setCoordinate(point1, data)
point1.setCentroids(point1, i_point1)
point1_final_centroids = point1.repeatKMeansClustering(point1)[0]
point1_total_cost = point1.repeatKMeansClustering(point1)[1]
print("Final Centroids: ", point1_final_centroids)
print("Total Cost: ", point1_total_cost)

# Visiaulize the set 1 cluster using matplotlib
point1_data = np.array(point1.coordinate)
point1_centroids = np.array(point1.centroids)
plt.scatter(point1_data[:,0],point1_data[:,1], c=point1.centroid_idx, cmap='rainbow')
plt.scatter(point1_centroids[:,0] ,point1_centroids[:,1], color='black')

# Set 2
point2 = KMeans
point2.setCoordinate(point2, data)
point2.setCentroids(point2, i_point2)
point2_final_centroids = point2.repeatKMeansClustering(point2)[0]
point2_total_cost = point2.repeatKMeansClustering(point2)[1]
print("Final Centroids: ", point2_final_centroids)
print("Total Cost: ", point2_total_cost)

# Visiaulize the set 2 cluster using matplotlib
point2_data = np.array(point2.coordinate)
point2_centroids = np.array(point2.centroids)
plt.scatter(point2_data[:,0],point2_data[:,1], c=point2.centroid_idx, cmap='rainbow')
plt.scatter(point2_centroids[:,0] ,point2_centroids[:,1], color='black')