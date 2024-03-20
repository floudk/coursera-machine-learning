# Regression Model
## Linger Regression Model
$ y = W x + B$  
W: parameter, weights  
B: parameter, bias

Basically, linger regression model will try to fit the known data(training set) to obtain kind of model,
then use the model to predict the new cases(data not in training set).

## Cost Function
$J(W,B) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 $  
$ \hat{y}^{(i)} = W x^{(i)} + B $: prediction  
$ y^{(i)} $: true value

Here, $\frac{1}{2m}$ is primarily for the convenience of the calculus, that is, when we take the derivative of the cost function, the $\frac{1}{2}$ will cancel out the 2 in the power of the cost function.


We try to minimize the cost function. In linear regression, the cost function $J(W,B)$ is kind of 3-D quadratic function, or more generally, a convex function. 

There is always a global minimum for the convex function, but it is not always easy to find the global minimum, especially when the number of parameters is large.

## Gradient descent

$w=w-\alpha \frac{\partial}{\partial w} J(w,b)$
$b=b-\alpha \frac{\partial}{\\partial b} J(w,b)$

Baed on matrix calculation, we can also find the optimal W and B by solving the normal equation, but this method only works in linear regression, and it is not efficient when the number of features is large.

Conversely, gradient descent is a more general method, and it is also more efficient when the number of features is large.