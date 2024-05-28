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

# Recommender systems

Simply speaking, recommender systems trying to predict how users will rate/judge several unobserved items before, and if the systems believe certain users will also like some new items based on their history behaviour then these items is recommended by the systems.

Typically, we need not only rating data but also other features to recommend. 
And it is actually a regression problem, where we describe the user $j$'s rating on movies by parameter $\omega^{(j)}$ and $b^{(j)}$, then the predicted rating is $\omega^{(j)} \cdot x^{(i)} + b^{(j)}$.
And the learning algorithm is to find the optimal $\omega^{(j)}$ and $b^{(j)}$ that minimize the cost function.

To learn all the parameters, we can define a global cost function that sums up the cost function for each user, including regularization terms to prevent overfitting.

## Collaborative filtering
However, when we don't have any features except for the rating data, how can we recommend items to users?

Collaborative filtering is a method that can simultaneously learn the features and the ratings parameters of the users and items.
Generally speaking, it is a method that considers both $x^{(i)}$ and $\omega^{(j)}, b^{(j)}$ as parameters to learn.
Therefore, the cost function is defined as the sum of the squared errors of the predicted ratings and the actual ratings, plus regularization terms of both $x^{(i)}$ and $\omega^{(j)}, b^{(j)}$.

Furthermore, when labels are binary, we can use logistic regression to predict the ratings that are binary, and hence the cost function is also using the logistic regression cost function, which also need to consider all the parameters of users and items.

## Implementational details

`Mean Normaliztion` is an useful tool when adding new items or new users, which actually substract the mean value and add the mean after predictions. By this way, we can use kind of average view to consider a new item or user.

## Content-based filtering

While collaborative filtering use rating of users who gave similar ratings as us, content-based filtering use features of user and items to find a good match.

And deep learning will be a good choice when we want to do content-based filtering, which can learn the features of users and items automatically.
Simply speaking, there are two neural networks, one for users and one for item features, and the output of the two networks are multiplied to get the predicted rating. During training, we still want to minimize the cost function, which is the sum of the squared errors of the predicted ratings and the actual ratings, plus regularization terms of both user and item features.

And when we want to do recommendation, we can use the trained neural networks to predict the ratings of users on items, and recommend the items with the highest predicted ratings. Also we can find similar items by calculating the similarity of the features of items.

### Recommending from a large dataset

Two steps: Retrieval and ranking.
Retrieval: find a small subset of items that are most likely to be recommended to the user, like use most similar items to the user's latest watched item.
Ranking: rank the items in the subset by the predicted ratings.

There also a trade-off between the retrieval performance and overhead, bigger subset will have better performance but also have more overhead.

## Principal Component Analysis

The motivation of PCA is to reduce the dimension of the data, which can be used for data compression, visualization, and speeding up learning algorithms.
Typically, PCA tries to find new axes and coordinates to represent the data, which can be used to reduce the dimension of the data.

Simply speaking, PCA uses projection to reduce the dimension of the data, and to make sure the information loss is minimized, PCA tries to make the projection as close to the original data as possible.
