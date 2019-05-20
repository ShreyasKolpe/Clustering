'''
K Means Clustering
Author - Shreyas Kolpe
Date - 2/28/2018
'''

import numpy as np 
import matplotlib.pyplot as plt
import random

# Calculated distance between two points
def distance(point1, point2):
	return np.linalg.norm(point1 - point2)

# Takes a single point and returns nearest of the cluster centroids
def assignNearest(point, centroids):
	minDist = float('Inf')
	minCluster = None
	for key in centroids.keys():
		dist = distance(point,centroids[key])
		if(dist < minDist):
			minDist = dist
			minCluster = key

	return minCluster

# Estimates the values of the centroids given points and their cluster membership
def recomputeCentroids(centroids, points, membership):
	# Code uses optimized vector operations in the form of logical arrays
	for key in centroids.keys():
		boolarr = membership == key
		if(np.count_nonzero(boolarr) != 0):
			centroids[key] = sum(points[boolarr])/np.count_nonzero(boolarr)

	return centroids		


# reading data
file = open('clusters.txt','r')
points = []
for line in file:
	values = line.split(',')
	points.append([float(values[0]), float(values[1])])

# storing data as numpy.ndarray
points = np.array(points, dtype=np.float64)

# array that holds cluster membership or labels for each point
membership = np.zeros(points.shape[0], dtype=np.int32)

# the given number of clusters
K = 3
# The numbe rof points and the nume=ber of dimensions
N, d = points.shape

centroids = {}
# data structures for coloring on the plot
labelColorMap = {}
# lambda function for getting a random value from 0-255
randColor = lambda: random.randint(0,255)
i = 1
while i <= K:
	# Initializing cluster centroids with randomly chosen data points
	centroids[i] = points[np.random.randint(0,N)]
	# Assigning a random color to this cluster
	labelColorMap[i] = '#%02X%02X%02X' % (randColor(),randColor(),randColor())
	i+=1



# The Expectation Maximization loop
# update checks for cluster assignment convergence
update = True
while update == True:
	update = False
	for i in range(points.shape[0]):
		cluster = assignNearest(points[i], centroids)
		if(cluster != membership[i]):
			membership[i] = cluster
			update = True

	centroids = recomputeCentroids(centroids, points, membership)


print("The clusters and centroids are")
for key in centroids.keys():
	print(key,centroids[key])

# Plotting
# label_color = [labelColorMap[cluster] for cluster in membership]
# fig = plt.figure('K Means')
# ax = plt.subplot(1,1,1)
# ax.set_xlabel('Feature x1')
# ax.set_ylabel('Feature x2')
# plt.scatter(points[:,0], points[:,1], c=label_color)
# plt.show()