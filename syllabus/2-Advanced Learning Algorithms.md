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
1. more training data
2. smaller sets of features
3. get more features
4. try adding polynomial features
5. decreasing/increasing learning rate


Cross-validation:  
Besides the training set and test set, we can also use the validation set to tune the hyperparameters.


## decision trees