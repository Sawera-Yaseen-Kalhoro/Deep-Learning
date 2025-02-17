"""
    Exercise 3: Linear Regression in PyTorch
"""

import torch
import torch.nn as nn
import torch.optim as optim
# from torchviz import make_dot

# Data Generation
N = 50  # Number of data points
X = torch.randn(N)  # Input data
Y = 2 * X + 3 + torch.randn(N) * 0.5  # Generating output data with noise


# parameter initialization
a = torch.randn(1, requires_grad = True)  # Allow PyTorch to compute gradients during backpropagation.
b = torch.randn(1, requires_grad = True) 


# optimization procedure: Gradient descent
optimizer = optim.SGD([a, b], lr=0.01)

for i in range(10):
    # Affine regression model
    Y_ = a * X + b

    diff = (Y - Y_)

    # Mean squared error (MSE) loss
    loss = torch.sum(diff**2) / N

    # Manual Gradient Calculation
    grad_a = sum(2 * (Y - Y_) * -X) / N
    print('Gradient w.r.t to a is : ',grad_a)
    
    grad_b = sum(2 * (Y - Y_) * -1) / N   
    print('Gradient w.r.t to b is : ',grad_b)

    # Gradient reset
    optimizer.zero_grad()

    # Gradient calculation
    loss.backward()

    # Visualize the computation graph
    # make_dot(Y_, params=dict(a=a, b=b)).render("linreg", format="png")

    # optimization step
    optimizer.step()
    print('Grad a: ',a.grad)
    print('Grad b: ',b.grad)

    # Compare the Gradient Calculations
    assert torch.allclose(a.grad, grad_a, atol=1e-5), 'Gradient w.r.t a is not correct'
    assert torch.allclose(b.grad, grad_b, atol=1e-5), 'Gradient w.r.t b is not correct'


    if i % 1000 == 0:
        # print(f'step: {i}, loss:{loss}')
        print(f'step: {i}, loss:{loss}, Y_:{Y_}, a:{a}, b {b}')

# print(f'a: {a}, b: {b}')