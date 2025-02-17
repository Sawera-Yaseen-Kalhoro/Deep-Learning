"""
    Exercise 5: Configurable deep models in PyTorch
    Feedforward neural network with customizable architecture and training
"""

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
from data import *

class PTDeep(nn.Module):
    def __init__(self, config, activation = torch.relu):  # Activation = ReLU/ Sigmoid
        super(PTDeep, self).__init__()

        self.layers = len(config) - 1
        self.weights = nn.ParameterList()
        for i in range(self.layers):
            weight = nn.Parameter(torch.randn(config[i], config[i + 1]))
            self.weights.append(weight)

        self.biases = nn.ParameterList()
        for i in range(self.layers):
            bias = nn.Parameter(torch.randn(1, config [i+1]))
            self.biases.append(bias)

        self.loss_images_num = 10
        self.most_loss_images = []
        self.activation = activation
        

    def forward(self, X):
        self.Y_ = X

        for i in range(self.layers):
            self.Y_ = torch.mm(self.Y_, self.weights[i]) + self.biases[i]
            
            if i != self.layers - 1:  # Last layer
                self.Y_ = self.activation(self.Y_)
            
            else:
                max_values , indices = torch.max(self.Y_, dim = 1)
                max_values = max_values.view(-1,1)
                self.Y_ = self.Y_ - max_values
                self.Y_ = self.Y_.double()
                self.prob = torch.softmax(self.Y_ , dim = 1)

        return self.prob


    def get_loss(self, X, Yoh_, param_lambda):
        vectorized_weights = torch.cat([self.weights[i].view(-1) for i in range(self.layers)])
        L2_norm = torch.norm(vectorized_weights, p = 2)
        true_classes = torch.argmax(Yoh_, dim=1)  
        neg_log_like = -torch.log(self.prob[range(self.prob.size(0)), true_classes])
        self.loss = torch.mean(neg_log_like) + (param_lambda * L2_norm)

        # Images which contribute the most to the loss
        # top_values, top_indices = torch.topk(torch.abs(neg_log_like), k = self.loss_images_num)
        top_values, top_indices = torch.topk(torch.abs(neg_log_like), k=min(self.loss_images_num, neg_log_like.size(0)))
        combined_list = []
        top_values_np = top_values.detach().numpy()
        top_indices_np = top_indices.detach().numpy()

        for loss, index in zip(top_values_np, top_indices_np):
            combined_list.append((loss, index))
            self.most_loss_images.extend(combined_list)

            unique_loss_images = {}
            for loss, index in self.most_loss_images:
                if index not in unique_loss_images:
                    unique_loss_images[index] = loss

            self.most_loss_images = [(loss, index) for index, loss in unique_loss_images.items()]
            self.most_loss_images.sort(key = lambda x: x[0], reverse = True)

            if len(self.most_loss_images) > self.loss_images_num:
                self.most_loss_images = self.most_loss_images[:self.loss_images_num]


def train(model, X, Yoh_, param_niter, param_delta, param_lambda, optimizer = torch.optim.SGD):
    """
    Arguments:
            - X: Model inputs [NxD], type : torch.Tensor
            - Yoh_: Ground truth [NxC], type : torch.Tensor
            - param_niter: Number of training iteration
            - param_delta: Learning rate 
            - param_lambda: Regularization term
            - Optimizer: Stochastic Gradient Descent
    """

    optimizer = optimizer(model.parameters(), lr = param_delta)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 1-1e-4)
    stored_loss = []

    # Training
    for i in range(param_niter):

        # Forward Pass
        model.forward(X)
        
        # Loss
        model.get_loss(X, Yoh_, param_lambda = param_lambda)
        stored_loss.append(model.loss.detach().numpy())
        
        # Reset Gradient
        optimizer.zero_grad()
        
        # Backward Pass
        model.loss.backward()
        
        # Parameter Update
        optimizer.step()

        if i % 1000 == 0 and i != 0:
            print(f'Iteration: {i}, loss: {model.loss}')

    optimizer.__class__.__name__ == 'SGD'
    
    # if optimizer.__class__.__name__ == 'Adam':
    #     scheduler.step()
    
    return stored_loss

def eval(model, X):
    """
    Arguments:
            - model: type: PTLogreg
            - X: actual datapoints[NxD], type: np.array
    """

    X_tensor = torch.Tensor(X)
    model.forward(X_tensor)
    return torch.Tensor.numpy(model.prob.detach())


# def decfun(model):
#     def classify(X):
#         return np.argmax(eval(model, X), axis=1)
#     return classify

def decfun(model):
    def classify(X):
        return model.forward(torch.Tensor(X)).detach().numpy()[:,1]
    return classify
        

def count_params(model):
    total_params = 0
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Size: {param.size()}")
        total_params += param.numel()  
    return total_params



if __name__ == "__main__":
    np.random.seed(100)

    # Define input data and labels
    # X, Y_ = sample_gauss_2d(3, 100)
    X, Y_ = sample_gmm_2d(6, 2, 10)
    Yoh_ = class_to_onehot(Y_)

    ptlr = PTDeep([2,10,2], torch.relu)

    # Parameter Learning
    train(ptlr, torch.Tensor(X), torch.Tensor(Yoh_), 10000, param_delta = 0.1, param_lambda = 1e-4)

    # Save the computational graph
    file_dirc = os.path.dirname(os.path.abspath(__file__))
    # Generate the graph
    make_dot(ptlr.prob, params = ptlr.state_dict()).render("pt_deep", directory = file_dirc, format = "png", cleanup = True)

    # Probabilities on Training data
    probs = eval(ptlr, X)
    Y = np.argmax(probs, axis = 1)

    average_precision = eval_AP(Y)
    accuracy, pr , M = eval_perf_multi(Y, Y_)
    print("Accuracy:", accuracy)
    for i, (recall, precision) in enumerate(pr):
         print("Class {}: Recall = {:.2f}, Precision = {:.2f}".format(i, recall, precision))
    print("Average Precision: {}". format(average_precision))
    print("Total number of parameters:", count_params(ptlr))


    # Plot Results
    decfun = decfun(ptlr)
    bbox = (np.min(X, axis = 0), np.max(X, axis = 0))
    graph_surface(decfun, bbox, offset = 0.5)
    graph_data(X, Y_, Y)
    plt.show()