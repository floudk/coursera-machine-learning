# Advanced Learning Algorithms

## Neural Network

Forward propagation for inference, and backward propagation for learning.

### Activation function
1. Sigmoid
2. ReLU

How to choose activation function?
- output layer:
    - Binary classification: Sigmoid
    - Regression: Linear activation function(y=+/-)
                  ReLU(y=0/+)

- Hidden layer: mostly use RuLU, which is faster than Sigmoid

Why we need activation function?

If no activation function, the nerual network is no more than linear regression, and thereby do not use linear function in hidden layers.

### Multiclass

$ a_j = \frac{e^{z_j}}{\sum_{k=1}^{N}e^{z_k}}$

Compared with logistic regression, we can expand the loss function to

$loss = -log{a_n}$

### Convolutional Layer

In the convolutional layer, each neuron is connected to a small region of the input image, and the weights are shared across different neurons.


### Applying ML

Training Notes:
1. more training data (to mitigate high variance)
2. smaller sets of features (to mitigate high variance)
3. get more features (to mitigate high bias)
4. try adding polynomial features (to mitigate high bias)
5. decreasing/increasing learning rate (to mitigate high bias/variance)

Cross-validation:  
Besides the training set and test set, we can also use the validation set to tune the hyperparameters.

**Bias** and **variance** are two essential concepts that help us understand the behavior of predictive models and why they might underperform
- Bias: refers to the error due to overly simplistic assumptions. 
        High Bias commonly means underfit, that is, both $J_{train}$ and $J_{cv}$ are high.
- Variance: describes the error due to too much complexity.
            High variance commonly means overfit, that is, low $J_{train}$ while high $J_{cv}$.

## Bias and Variance

### Regularization

To prevent the model from being too complicited, regularization item is added to Loss function:
$$\frac{\lambda}{2m}\sum_{j=1}{n}w_j}^2$$

However, how to decide a suitable $\lambda$ is not easy. If $\lambda$ is too high, the model may be too simple to fit well, or otherwise, it is too low and causes overfitting.

**Cross-Validation** also can be used to deterimine $\lambda$, that is, split the dataset into three parts: training set, validation set, and test set. Then, we can train the model with different $\lambda$ and choose the one with the lowest $J_{cv}$.

Typically, bigger $\lambda$ will lead to a simpler model, thereby higher $J_{train}$.
And for $J_{cv}$, typically, it will decrease first and then increase, which means the model is too complicated (overfitting) at first, and then too simple (underfitting).

- L1 regularization: $J_{reg} = \frac{\lambda}{2m}\sum_{j=1}{n}|w_j|$
  L1 regularization encourages sparsity, thereby can be used for both feature selection and model simplification.

- L2 regularization: $J_{reg} = \frac{\lambda}{2m}\sum_{j=1}{n}w_j}^2$
  L2 regularization will decrease the weights but not to zero, thereby can be used for model simplification but not feature selection.

### Baseline

Common baseline includes:
- Human-level performance
- SOTA model/algorithm performance
- Guass Based on Experience

### Learning Curves

Learning curves are a good way to tell if the model is overfitting or underfitting.
Suppose we have training size-Error curve about both training_error and cv_error:
- if the model is underfitting, both training_error and cv_error will first decrease and then converge to a high value. Because the model is too simple, and it is useless to add more data.
- if the model is overfitting, both training_error and cv_error will first decrease and then diverge, but adding more data will help to decrease cv_error until it converges to a certain value.

To make it simple, more data needs relatively more complex models.

### Development Loop

Development Loop: choose architecture(data,model), train, diagnostics

### Data-Centric Way

AI = Model + Data

Traditional way is model/algorithms-centric

For data-centric way, adding more data, data augmentation, generating synthetic data, and **transfer learning** are all good ways.

Tranfer learning: use the pre-trained model on a similar task to help the current task.
Typically, we can obtain the pre-trained model from other tasks with a large dataset or well-recoginized pre-trained models. And then, we can use the pre-trained model to do the current task, but the last layer may need to be re-built in different shapes.

And for the fine-tuning, there are two ways:

1. Freeze all layers except the last layer, and then train the last layer.
2. Train all layers, but with a smaller learning rate.

Typically, the first way is more common for a small dataset, and the second way is more common for a large dataset.

### Skewed Data

Precision: $\frac{TP}{TP+FP}$, means the proportion of positive identifications that were actually correct.
Recall: $\frac{TP}{TP+FN}$, means the proportion of actual positives that were identified correctly.

The presure of high precision and high recall is kind of trade-off, since high confidence will lead to high precision but low recall, and vice versa.

Therefore, **F1 score** is introduced to balance the precision and recall:

$$F1 = \frac{2}{\frac{1}{precision}+\frac{1}{recall}}=\frac{2*precision*recall}{precision+recall}$$

F1 score is the **harmonic mean** of precision and recall, and it is a good way to balance the precision and recall.




## decision trees

The learning process in decision trees are about the following questions:

1. How to choose features to split on each node?
- For **maximize purity**, that is, for nodes closer to root nodes, the feature can gain hign purity/information gain. And this is a recursive algorithm to build trees.

2. When to stop splitting?
- When a node is 100% one class
- When splitting a node will result in the tree excedinig a maximum depth
- When improvements in purity score are below a threshold
- When number of examples in a node is below a threshold

`Purity Quantity`, we can define impurity: Let $p_1=$ fraction of examples that are not some classfication(two-classfication case), then impurity $H(p_1) = -p_1 \log_2(p_1) - p_0\log_2(p_0)$

use **one-hot encoding**: use k binary features to replace a k-value `categorical feature`.

use threshold to split `Continuous valued features`to 2-cases, that is, also kind of binary feature.

### Regression Trees

For regression task with decision tree, what we want to is predict a number, so when spliting, we want to all objects in a class is as close as possible, that is, **variance**.
And define impurity with variance, we can do exactly what classification task does.

### Tree ensembles

decision trees are sentitive to small changes in dataset, so it is better to train a lot of trees together, that is **ensemble tree**, and use their predictions vote for a final one.

*Samping with replacement* means contructing trees with sampling and replacing these into the whole dataset before next sampling.

**Random forest** algorithm: not only select sampling in dataset, but also use sampling in features, that is use subset of features to train for a more robust prediction.

### Boosted trees

Intuition: instead of sampling with equal probability, make it more likely to pick misclassified examples from previously trained trees.

**XGBoost** means eXtream Gradient Boosting, is an open source fast efficient implementation of boosted trees, which has a good choice of default splitting criteria and criteria for when to stop splitting and built=in regularization.

### Decision Tree vs Neural Networks

Decision Tree:
- work well on tabular(structured) data
- Not suitable for unstructured data
- Fast
- Small decision trees may be human interpretable

### Neural Networks
- work well on all types of data, including tabular and unstructured data.
- may be slower than a decision tree.
- work with transfer learning
- easier to string together multiple neural networks.