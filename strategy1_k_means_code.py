# from Precode import *
import numpy as np
import random
from scipy.io import loadmat
import matplotlib.pyplot as plt
# data = np.load('AllSamples.npy')
data = loadmat('project_allSamples.mat')['AllSamples']

# functions to generate k values and their initial points
def initial_point_idx(id, k,N):
    return np.random.RandomState(seed=(id+k)).permutation(N)[:k]

def init_point(data, idx):
    return data[idx,:]

def initial_S1(id):
    print("Strategy 1: k and initial points")
    i = int(id)%150 
    random.seed(i+500)
    k = [x for x in range(2, 11)]
    k_points = []
    
    for num in range(len(k)):
        init_idx = initial_point_idx(i,k[num],data.shape[0])
        init_s1 = init_point(data, init_idx)
        k_points.append(init_s1)
    return k, k_points

stg1_k_values, stg1_k_points = initial_S1('0233')

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

# loss functions for strategy 1 to calculate the loss for each k
def lossFunctionStg1(stg1_k_values, stg1_k_points):
    loss = []
    for i in range(len(stg1_k_values)):
        kmeans = KMeans_stg1 
        kmeans.setCoordinate(kmeans, data)
        kmeans.setK(kmeans, stg1_k_values[i])
        kmeans.setCentroids(kmeans, stg1_k_points[i])
        print("k value: ", stg1_k_values[i])
        final_centroids, total_cost = kmeans.repeatKMeansClustering(kmeans)
        print("final centroids: ", final_centroids)
        print("loss: ", total_cost)
        loss.append(total_cost)
    plt.scatter(stg1_k_values, loss)
    plt.xlabel("k value")
    plt.ylabel("Loss")
    plt.show()

lossFunctionStg1(stg1_k_values, stg1_k_points)