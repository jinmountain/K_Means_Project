# from Precode import *
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
# data = np.load('AllSamples.npy')
data = loadmat('project_allSamples.mat')['AllSamples']
# k1,i_point1,k2,i_point2 = initial_S1('0233') # please replace 0111 with your last four digit of your ID
# the given k and initial centroids for each set are below
k1 = 3
i_point1 = np.array([[ 7.78551305, 3.12724529], [ 3.72610844, 5.20432439], [3.2492998, 5.59125171]])
k2 = 5
i_point2 = np.array([[ 2.3537231, 6.29810755], [ 2.81629029, 3.1999725 ], [ 6.6161895, 0.66750633], [ 5.38398051, 3.53840433], [ 7.59731342, 1.16504743]])

# Assign the initial random centroids from the points
class KMeans_stg1:
    def __init__(self):
        self.coordinate
        self.centroids
        self.centroid_idx
        self.cost

    # Set the samples
    def setCoordinate(self, coordinate):
        self.coordinate = coordinate
    
    # Set the k given
    def setK(self, k):
        self.k = k
    
    # Set the centroids
    def setCentroids(self, centroids):
        self.centroids = centroids
        
    # function to find the closest centroid for each sample
    def closestCentroid(self):
        min_centroid_idx_list = []
        min_cost_list = []
        # loop through all the samples
        for c in range(len(self.coordinate)):
            min_centroid_idx = 0
            min_cost = float('inf')
            
            # calculate the loss from each centroid and find which centroid is the closest
            for i in range(self.k):
                x = self.coordinate[c][0] - self.centroids[i][0]
                y = self.coordinate[c][1] - self.centroids[i][1]
                loss = x**2 + y **2
                if min_cost > loss:
                    min_centroid_idx = i 
                    min_cost = loss
            # add the closest centroid's index and cost
            min_centroid_idx_list.append(min_centroid_idx)
            min_cost_list.append(min_cost)

        self.centroid_idx = min_centroid_idx_list
        self.cost = min_cost_list
    
    # Find the mean centroid for each cluster
    def clusterMeanCentroid(self):
        # counts the number of point to each centroid in popular_centroid
        popular_centroids = [0 for i in range(self.k)]
        # sum of the points close to each centroid
        centroids = [[0,0] for i in range(self.k)]
        
        # To find the mean centroid of each cluster
        # add all the x and y value of  each sample to its centroid
        for i in range(len(self.coordinate)):
            centroids[self.centroid_idx[i]][0] += self.coordinate[i][0]
            centroids[self.centroid_idx[i]][1] += self.coordinate[i][1]
            # +1 for each centroid when a sample belongs to it
            popular_centroids[self.centroid_idx[i]] += 1

        # divide by the number of samples in each cluster to get mean
        for i in range(self.k):
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
        
        # when pre centroids and new centroids aren't same
        # keep looping until they are same 
        while pre_centroids != new_centroids:
            num_repeat += 1

            pre_centroids = new_centroids
            self.closestCentroid(self)
            self.clusterMeanCentroid(self)
            new_centroids = self.centroids

        # loop stopped and sum up the cost
        final_cost = sum(self.cost)

        return [self.centroids, final_cost]

print("K Means Strategy 1 Set 1")
point1 = KMeans_stg1
point1.setCoordinate(point1, data)
point1.setK(point1, k1)
print("k: ", point1.k)
print("Initial points assigned: ", i_point1)
point1.setCentroids(point1, i_point1)
print(point1.centroids)

point1_final_centroids = point1.repeatKMeansClustering(point1)[0]
point1_total_cost = point1.repeatKMeansClustering(point1)[1]
print("Final Centroids: ", point1_final_centroids)
print("Total Cost: ", point1_total_cost)

# Visiaulize the cluster using matplotlib
# Green points are the first k centroids and lack points are the last k centroids
point1_data = np.array(point1.coordinate)
point1_centroids = np.array(point1.centroids)
plt.scatter(point1_data[:,0],point1_data[:,1], c=point1.centroid_idx, cmap='rainbow')
plt.scatter(point1_centroids[:,0] ,point1_centroids[:,1], color='black')
plt.scatter(i_point1[:,0] ,i_point1[:,1], color='green')
plt.show()

print("K Means Strategy 1 Set 2")
point2 = KMeans_stg1
point2.setCoordinate(point2, data)
point1.setK(point2, k2)
print("k: ", point2.k)
print("Initial points assigned: ", i_point2)
point2.setCentroids(point2, i_point2)
point2_final_centroids = point2.repeatKMeansClustering(point2)[0]
point2_total_cost = point2.repeatKMeansClustering(point2)[1]
print("Final Centroids: ", point2_final_centroids)
print("Total Cost: ", point2_total_cost)

# Visiaulize the cluster using matplotlib
# Green points are the first k centroids and lack points are the last k centroids
point2_data = np.array(point2.coordinate)
point2_centroids = np.array(point2.centroids)
plt.scatter(point2_data[:,0],point2_data[:,1], c=point2.centroid_idx, cmap='rainbow')
plt.scatter(point2_centroids[:,0] ,point2_centroids[:,1], color='black')
plt.scatter(i_point2[:,0] ,i_point2[:,1], color='green')
plt.show()