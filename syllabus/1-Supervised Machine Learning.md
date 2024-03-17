# Regression Model

## Linger Regression Model
$ y = W x + B$  
W: parameter, weights  
B: parameter, bias

## Cost Function
$J(W,B) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 $  
$ \hat{y}^{(i)} = W x^{(i)} + B $: prediction  
$ y^{(i)} $: true value

Here, $\frac{1}{2m}$ is primarily for the convenience of the calculus, that is, when we take the derivative of the cost function, the $\frac{1}{2}$ will cancel out the 2 in the power of the cost function.


We try to minimize the cost function. In linear regression, the cost function $J(W,B)$ is a quadratic function, or more generally, a convex function.   
