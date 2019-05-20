'''
Gaussian Mixture Models
Author - Shreyas Kolpe
Date - 2/28/2018
'''

import numpy as np 
from scipy.stats import multivariate_normal
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import random

# An instance of this class refers to a Gaussian parametrized by mu, sigma and pi
class Gaussian(object):
	'An instance refers to a single Gaussian distribution'
	def __init__(self, mean, amplitude, covariance):
		self.mean = mean
		self.amplitude = amplitude
		self.covariance = covariance
		self.normal = multivariate_normal(mean=self.mean, cov=self.covariance)

	def pdf(self, X):
		return (self.amplitude)*self.normal.pdf(X)

# Estimates the Gaussian parameters for the K Gaussians given 
def computeGaussians(points, membership, K, N, d):
	# calculate the K means
	means = membership.transpose().dot(points)
	means = means/membership.sum(axis=0)[:,None]
	
	gaussians = []

	# calculating the K amplitudes
	amplitude = membership.sum(axis=0)/N
	
	X = []
	sigma = []
	# mean standardizing the data and initializing covariance matrices
	for i in range(K):
		X.append(points - means[i])
		sigma.append(np.zeros((d,d)))

	# calculating covariance matrices as weighted sum of outer products of mean standardized data vectors
	for i in range(N):
		for j in range(K):
			sigma[j] = sigma[j] + membership[i,j]*(np.outer(X[j][i], X[j][i]))

	# normalizing factor for covariance matrices
	for i in range(K):
		sigma[i] = sigma[i]/membership.sum(axis=0)[i]
		gaussians.append(Gaussian(means[i], amplitude[i], sigma[i]))

	return gaussians

# calculates log likelihood that the given Gaussians explain the data points
# It will be used as a test for convergence
def likelihood(gaussians, points):
	values = np.empty((N,K))
	for i in range(K):
		values[:,i] = gaussians[i].pdf(points)
		
	logLikelihood = np.log(values.sum(axis=1)).sum()
	return logLikelihood

# Update membership values based on Gaussian parameters estimated
def computeMembership(gaussians, points, membership, N, K):
	newMembership = np.empty((N,K))
	for i in range(K):
		newMembership[:,i] = gaussians[i].pdf(points)
		
	newMembership = newMembership/newMembership.sum(axis=1)[:,None]
	return newMembership

# Initialize weights/memmberships randomly but normalized per cluster
def initializeWeights():
	membership = np.random.rand(N,K)
	normalizer = membership.sum(axis=1)
	membership = membership/normalizer[:,None]	
	return membership

# method to draw the ellipses representing the Gaussian
def draw_ellipses(gaussians, ax, labelColorMap):
	index=0
	for curve in gaussians:
		v, w = np.linalg.eigh(curve.covariance)
		u = w[0] / np.linalg.norm(w[0])
		angle = np.arctan2(u[1], u[0])
		angle = 180 * angle / np.pi  # convert to degrees
		v = 2. * np.sqrt(2.) * np.sqrt(v)
		ell = mpl.patches.Ellipse(curve.mean, v[0], v[1], 180 + angle, color=labelColorMap[index])
		ell.set_clip_box(ax.bbox)
		ell.set_alpha(0.5)
		ax.add_artist(ell)
		index+=1

# read data
file = open('clusters.txt','r')
points = []
for line in file:
	values = line.split(',')
	points.append([float(values[0]), float(values[1])])
file.close()


points = np.array(points)

N, d = points.shape
K = 3

# randomly intialize memberships
membership = initializeWeights()
# label colors
labelColorMap = {}
randColor = lambda: random.randint(0,255)
for i in range(K):
	labelColorMap[i] = '#%02X%02X%02X' % (randColor(),randColor(),randColor())

# Let initial log likelihood be 0. 
logLikelihood = 0.0
# The threshold to cut off EM loop
threshold = 0.000000001
while True:
	# compute Gaussian parameters
	gaussians = computeGaussians(points, membership, K, N, d)
	# compute current log likelihood
	current_ll = likelihood(gaussians, points)
	# break loop if difference between current and previous likelihood smaller than threshold
	if(abs(current_ll - logLikelihood) < threshold):
		break
	logLikelihood = current_ll
	# update membership values
	membership = computeMembership(gaussians, points, membership, N,K)

# find cluster to which point belongs as the one with maximum membership value
maxMembership = np.argmax(membership, axis=1)

# printing parameters
index = 0
for curve in gaussians:
	print("Gaussian",i)
	print("Mean :",curve.mean)
	print("Amplitude :",curve.amplitude)
	print("Covariance Matrix :")
	print(curve.covariance)
	print("------------------")
	index+=1

# plotting
label_color = [labelColorMap[cluster] for cluster in maxMembership]
fig = plt.figure('GMM')
ax = plt.subplot(1,1,1)
ax.set_xlabel('Feature x1')
ax.set_ylabel('Feature x2')
plt.scatter(points[:,0], points[:,1], c=label_color)
draw_ellipses(gaussians,ax,labelColorMap)
plt.show()