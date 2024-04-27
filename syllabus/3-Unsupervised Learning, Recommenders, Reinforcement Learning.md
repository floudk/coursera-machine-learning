# Clustering

## K-means algorithm

1. Randomly initialize K cluster centroids
2. Recompute cluster centroids:
    - for each point, assign it to the nearest centroid: $c^{(i)} = \text{argmin}_k ||x^{(i)} - \mu_k||^2$
    - for each centroid, set it to the mean of the points assigned to it: $\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x^{(i)}$
3. Repeat step 2 until convergence or meet some stopping criteria

### Optimization objective

Cost function: $J(c, \mu) = \frac{1}{m} \sum_{i=1}^m ||x^{(i)} - \mu_{c^{(i)}}||^2$

### Initialization

pick K training examples at random and set $\mu_1, \mu_2, \ldots, \mu_K$ equal to these K examples. 

And it is better to run K-means multiple times with different random initializations and pick the one that gives the lowest cost function value.

### Choosing the number of clusters

Elbow method: plot cost function $J$ as a function of the number of clusters $K$ and look for the "elbow" point.

elbow point: the point where the cost function value starts to decrease more slowly.

# Anomaly detection

Anomaly detection is used when we have a small number of positive examples (anomalies) and a large number of negative examples. 
We want to learn a model that can detect the anomalies, which may represent a defect in a manufacturing process, a credit card fraud, etc.

## Density estimation

Given a training set $\{x^{(1)}, x^{(2)}, \ldots, x^{(m)}\}$, we need to build a model $p(x)$ that estimates the probability of a new example being normal.

### Gaussian distribution

Given a training set $\{x^{(1)}, x^{(2)}, \ldots, x^{(m)}\}$, we can estimate the Gaussian distribution for each feature $x_i$.
Then, the probability of a new example $x$ can be calculated based on the pre-computed Gaussian distribution.

And for anomaly detection with multiple features, we can assume that the features are independent and calculate the probability as follows:
$p(x) = p(x_1; \mu_1, \sigma_1^2) \times p(x_2; \mu_2, \sigma_2^2) \times \ldots \times p(x_n; \mu_n, \sigma_n^2)$. Then compare $p(x)$ with a threshold $\epsilon$ to decide if it is an anomaly.

## Developing and evaluating

Although the anomaly detection handles mainly unlabeled data, some labeled data are still useful for evaluating and developing this kind of system.

And for anomaly detection vs. supervised learning, generally, we should use anomaly detection when we have a small number of positive examples and a large number of negative examples, and use supervised learning when we have a large number of positive examples and a large number of negative examples.
Furthermore, anomaly detection can handle novel anomalies that are never seen before, while supervised learning cannot.

When choosing features for anomaly detection, we should refactoring the features to make them more Gaussian-like to improve the performance of the Gaussian distribution model. Also, we can use multiple features to calculate a new feature that is more Gaussian-like.