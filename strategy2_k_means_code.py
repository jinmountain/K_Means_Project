# from Precode2 import *
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
# data = np.load('AllSamples.npy')
data = loadmat('project_allSamples.mat')['AllSamples']
# k1,i_point1,k2,i_point2 = initial_S1('0233') # please replace 0111 with your last four digit of your ID
# the given k and initial centroids for each set are below
k1 = 4
i_point1 = np.array([6.12393256, 5.49223251])
k2 = 6
i_point2 = np.array([ 3.2115245, 1.1089788])

# Assign the initial random centroids from the points
class KMeans_stg2:
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

    # Start with initial random centroid and find other i-1 centroids with maximum distance
    def findMaxDistCentroid(self, k, first_random_c):
        potential_centroids = self.coordinate
        # k slots for potential initial centroids
        init_centroids = [[0, 0] for k in range(k)]
        init_centroids[0] = first_random_c
        
        total_k = k
        # we start from k1 since we have k0 and k1
        # we use the distance of k0 and k1 to get k2 which is the avg farthest point from k0 and k1
        current_k = 1

        # loop until we have k initial centroids
        while current_k < total_k:
            max_dist = 0
            idx = 0
            for i in range(len(potential_centroids)):
                total_dist = 0

                for k in range(current_k):
                    # use uclidean distance
                    dist = np.sqrt((init_centroids[k][0] - potential_centroids[i][0])**2 + (init_centroids[k][1] - potential_centroids[i][1])**2)
                    total_dist += dist
                avg_total_dist = total_dist/(current_k)
                # if the average is distance is bigger 
                if max_dist < avg_total_dist:
                    max_dist = avg_total_dist
                    idx = i
                    
            # add the farthest sample to potential centroids
            init_centroids[current_k] = potential_centroids[idx]
            potential_centroids = np.delete(potential_centroids, idx, 0)
            current_k += 1
            
        return init_centroids

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

print("K Means Strategy 2 Set 1")
point1 = KMeans_stg2
point1.setCoordinate(point1, data)
point1.setK(point1, k1)
print("k: ", point1.k)
print("Initial point assigned: ", i_point1)
initial_centroids1 = np.array(point1.findMaxDistCentroid(point1, k1, i_point1))
print("Initial point assigned and other k-1 centroids: ", initial_centroids1)

point1.setCentroids(point1, initial_centroids1)
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
plt.scatter(initial_centroids1[:,0] ,initial_centroids1[:,1], color='green')
plt.show()

print("K Means Strategy 2 Set 2")
point2 = KMeans_stg2
point2.setCoordinate(point2, data)
point1.setK(point2, k2)
print("k: ", point2.k)
print("Initial point assigned: ", i_point2)
initial_centroids2 = np.array(point2.findMaxDistCentroid(point2, k2, i_point2))
print("Initial point assigned and other k-1 centroids: ", initial_centroids2)

point2.setCentroids(point1, initial_centroids2)
point2_final_centroids = point2.repeatKMeansClustering(point2)[0]
point2_total_cost = point2.repeatKMeansClustering(point2)[1]
print("Final Centroids: ", point2_final_centroids)
print("Total Cost: ", point2_total_cost)

point2_data = np.array(point2.coordinate)
point2_centroids = np.array(point2.centroids)
plt.scatter(point2_data[:,0],point2_data[:,1], c=point2.centroid_idx, cmap='rainbow')
plt.scatter(point2_centroids[:,0] ,point2_centroids[:,1], color='black')
plt.scatter(initial_centroids2[:,0] ,initial_centroids2[:,1], color='green')
plt.show()