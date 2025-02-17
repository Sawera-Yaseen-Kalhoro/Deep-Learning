"""
    Exercise 4: Logistic Regression in PyTorch
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
from data import *

class PTLogreg(nn.Module):
  def __init__(self, D, C):

    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
    """
    
    super(PTLogreg, self).__init__()
    self.W = nn.Parameter(torch.randn(D, C))
    self.b = nn.Parameter(torch.randn(1, C))

  
  def forward(self, X):
    self.Y_ = torch.mm(X, self.W) + self.b
    self.prob = torch.softmax(self.Y_, dim=1)

  
  def get_loss(self, X, Yoh_, param_lambda):
    vectorized_weights = self.W.view(-1)  # Flatten the weight tensor
    L2_norm = torch.norm(vectorized_weights, p=2)  # Calculate the L2 norm of the weights
    neg_log_like = torch.sum(-torch.log(self.prob[Yoh_ > 0])) / X.shape[0] # Negative log likelihood
    self.loss = neg_log_like + 0.5 * param_lambda * L2_norm  # Combined loss and regularization


def train(model, X, Yoh_, param_niter, param_delta, param_lambda):
  """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Yoh_: ground truth [NxC], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
     - param_lambda: Regularization term
  """

  optimizer = optim.SGD([model.W, model.b], lr = param_delta) # SGD to update the model parameters (W, b)

  # Training loop
  for i in range(param_niter):
    
    # Forward pass
    model.forward(X)
    
    # Loss
    model.get_loss(X, Yoh_, param_lambda)
    
    # Gradient reset
    optimizer.zero_grad()  # Clear any previous gradients
    
    # Backward pass
    model.loss.backward()
    
    # Parameter update
    optimizer.step() # Update the model parameters

    if i % 1000 == 0:
      print(f'Iteration: {i}, loss: {model.loss}')


def eval(model, X):

  """Arguments:
     - model: type: PTLogreg
     - X: actual datapoints [NxD], type: np.array
     Returns: predicted class probabilites [NxC], type: np.array
  """
  
  X_tensor = torch.Tensor(X)  # Convert input to tensor
  model.forward(X_tensor)     # Get predictions
  return torch.Tensor.numpy(model.prob.detach())  # Convert probabilities to Numpy array


def decfun(model):
    def classify(X):
      return np.argmax(eval(model, X), axis=1)  # Get the predicted class labels
    return classify


if __name__ == "__main__":
    # initialize random number generator
    np.random.seed(100)

    # define input data X and labels Yoh_
    X, Y_ = sample_gauss_2d(2, 100)
    # X, Y_ = sample_gmm_2d(4, 2, 40)
    Yoh_ = class_to_onehot(Y_)
    
    # define the model:
    ptlr = PTLogreg(X.shape[1], Yoh_.shape[1])

    # learn the parameters (X and Yoh_ have to be of type torch.Tensor):
    train(ptlr, torch.Tensor(X), torch.Tensor(Yoh_), 10000, 0.5, 0.01)

    # get probabilites on training data
    probs = eval(ptlr, X)
    Y = np.argmax(probs, axis=1)
    
    # print out the performance metric (precision and recall per class)
    accuracy, pr , M = eval_perf_multi(Y, Y_)
    print("Accuracy:", accuracy)
    for i, (recall, precision) in enumerate(pr):
        print("Class {}: Recall = {:.2f}, Precision = {:.2f}".format(i, recall, precision))


    # visualize the results, decicion surface
    decfun = decfun(ptlr)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(decfun, bbox, offset=0.5)
    graph_data(X, Y_, Y)
    plt.show()