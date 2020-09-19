# from Precode2 import *
import numpy as np
import random
from scipy.io import loadmat
import matplotlib.pyplot as plt
# data = np.load('AllSamples.npy')
data = loadmat('project_allSamples.mat')['AllSamples']

# functions to generate k values and their initial points
def initial_point_idx2(id,k, N):
    random.seed((id+k))     
    return random.randint(0,N-1)

def initial_S2(id):
    print("Strategy 2: k and initial points")
    i = int(id)%150 
    random.seed(i+800)
    k = [x for x in range(2, 11)]
    k_points = []
    
    for num in range(len(k)):
        init_idx = initial_point_idx2(i, k[num],data.shape[0])
        init_s2 = data[init_idx,:]
        k_points.append(init_s2)
    return k, k_points

stg2_k_values, stg2_k_points = initial_S2('0233')

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

def lossFunctionStg2(stg2_k_values, stg2_k_points):
    total_loss = []
    total_final_centroids = []
    for i in range(len(stg2_k_values)):
        kmeans = KMeans_stg2 
        kmeans.setCoordinate(kmeans, data)
        kmeans.setK(kmeans, stg2_k_values[i])
        initial_centroids = np.array(kmeans.findMaxDistCentroid(kmeans, stg2_k_values[i], stg2_k_points[i]))
        kmeans.setCentroids(kmeans, initial_centroids)
        print("k value: ", stg2_k_values[i])
        final_centroids, loss = kmeans.repeatKMeansClustering(kmeans)
        print("final centroids: ", kmeans.centroids)
        total_final_centroids.append(final_centroids)
        print("loss: ", loss)
        total_loss.append(loss)
    plt.scatter(stg2_k_values, total_loss)
    plt.xlabel("k value")
    plt.ylabel("Loss")
    plt.show()
    
lossFunctionStg2(stg2_k_values, stg2_k_points)