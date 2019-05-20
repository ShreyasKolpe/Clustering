# Clustering

Clustering is one of the broad category of unsupervised learning algorithms, along with dimensionality reduction and representation learning, in general. It is aimed at finding similarities in data when no labels are provided.
Here, the task is to implement K Means and GMMs to find groups in two dimensional data.


## K Means

The program `kmeans.py` is structured as follows –
* Method to calculate Euclidean distance between two points – `distance(point1, point2)` – return the L2 norm of the difference of the vectors (`numpy` `ndarray`) that represent the two points.

* Method to assign a given point to the cluster with the nearest cluster center – `assignNearest(point, centroids)`- returns the integer ID of the cluster whose center is nearest. The argument `centroids` is a `Python` `dictionary` that has the integer cluster ID as the key and the cluster center as value (`numpy` `ndarray`).

* Method to recompute cluster centroids when cluster membership gets updated – `recomputeCentroids(centroids, points, membership)` – return argument `centroids` updated by calculating the mean of all the points assigned to each cluster in membership. Some basic code optimization has been done in the form of using logical arrays to quickly index all points belonging to a cluster in the calculations.

* The code that loads the data, randomly initializes K cluster centers and performs Expectation Maximization.
Here, the data from `clusters.txt` is parsed, and stored in points – a `numpy` `ndarray` ofshape(N,d),where N is the number of points (150) and d is the number of dimensions (2).
`membership` is initially a vector of zeros.
The `centroids` dictionary is initialized for K clusters from among the data points chosen randomly, along with `labelColorMap` – a dictionary with randomly assigned hexadecimal color strings for each cluster.
Then, a loop implements Expectation Maximization by calling `assignNearest()` and `recomputeCentroids()` in turn, each using the output of the most recent call to the other. The variable `update` is used to break out of the EM loop when there is no change in cluster assignment between two successive loops.
Finally the cluster centers are printed and data plotted using the API provided by `matplotlib`.

![](https://github.com/ShreyasKolpe/Clustering/blob/master/k-means-visualization.png)

## Gaussian Mixture Models

GMMs allow a point to belong to different 'clusters' to varying degrees. More precisely, GMMs attempt to approximate a generative model that explains the data and this model is assumed to be a mixture of Gaussians. Finding the Gaussians involves finding their means, covariance matrices and amplitudes by the iterative Expectation Maximization algorithm.

The code `gmm.py` is organized as follows -
* The class `Gaussian` that represents a Gaussian parameterized by mean, covariance matrix and amplitude.
Its members `mean` and `covariance` are `numpy` `ndarrays` representing the mean and covariance respectively. The height of the Gaussian is represented by `amplitude`, while `normal` is an object of type `multivariate_normal` from `scipy` which is used to return the probability distribution function for given data point(s).
The constructor initializes all these members while `pdf(X)` returns the pdf for the data point(s) X.

* Method to calculate Gaussian parameters given points and their (soft) memberships –
`computeGaussians(points, membership, K, N, d)` – returns a `list` of objects of `Gaussian` type.
This method first calculates the means as weighted sum of points normalized by the sum of weights for the points belonging to that Gaussian. Wherever possible, the code is vectorized for speed. The amplitude is similarly the sum of weights for all points belonging to the Gaussian normalized by N – the number of points. X is a list of ndarray holding mean standardized data for the K Gaussians. It is used in calculating the covariance matrices sigma as weighted sum of outer products of vectors representing each point. It is also normalized.
Then each Gaussian is packed into the object of the same name and the list is
returned.
* Method to calculate log likelihood –
`likelihood(gaussians, points)` – returns a single float value that encapsulates how well the given Gaussians explain the data points. For a single point, the log likelihood is the log(sum of pdfs of the point for each Gaussian). The likelihood for the entire set of data points is the sum of likelihood over all points.

* Method to compute membership values from Gaussians -
`computeMembership(gaussians, points, membership, N, K)` – returns the updated `membership` as an `ndarray` using `gaussians` and `points`, as pdf of the data point in the Kth Gaussian normalized.
* Method to initialize membership/weights–
`initializeWeights()` –returns initial `membership` by filling with random values and normalizing per point.
* Method to plot ellipses representing the Gaussians – `draw_ellipses(gaussians, ax, labelColorMap)` – Plots in `ax` subplot using colors in `labelColorMap`.
* The code then reads in the data, storing points as an `ndarray`, calling `initializeWeights()` and the EM loop.

    The EM loop calls `computeGaussians()` and `computeMembership()` in turn. The loop terminates when the difference in likelihood values between two successive iterations of the loop is less than `threshold`.
The `maxMembership` is the label for each point as the Gaussian that gives it the highest membership value
Finally, the Gaussians are printed and data is plotted.

![](https://github.com/ShreyasKolpe/Clustering/blob/master/gmm-visualization2.png)

