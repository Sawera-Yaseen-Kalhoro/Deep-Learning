import numpy as np
import matplotlib.pyplot as plt
from data import *


def binlogreg_train(X, Y_ , param_niter = 100000, param_delta = 0.05):
    """
    Train a binary logistic regression model.
        Arguments:
            X: np.array of shape (N, D), features
            Y_: np.array of shape Nx1, Class labels

            Returns:
                w, b: Parameters of the binary logistic regression model
    """
    
    N, D = X.shape
    w = np.random.randn(D)
    b = 0
    
    for i in range(param_niter):
        # Classification scores
        scores = np.dot(X,w) + b
        
        # Calculate the probability of class 1
        prob = 1/(1+np.exp(-np.dot(X,w)-b))
        
        # calculate the loss
        loss = -np.sum(Y_*np.log(prob) + (1-Y_)*np.log(1-prob))
        
        # Trace the loss
        if i % 1000 == 0:
            print("iteration {}: loss {}".format(i, loss))
        
        # calculate the gradient
        grad_w = np.dot(X.T, prob - Y_)
        grad_b = np.sum(prob - Y_)
        
        # update the parameters
        w = w - param_delta * grad_w
        b = b - param_delta * grad_b
    
    return w, b

def binlogreg_classify(X, w, b):
    """
    Classify the data using the binary logistic regression model.
        Arguments:
            X: np.array of shape (N, D), features
            w, b: Parameters of the binary logistic regression model

            Returns:
                Y: np.array of shape Nx1, Class labels
    """
    
    # Classification scores
    scores = np.dot(X,w) + b
    
    # Calculate the probability of class 1
    prob = 1/(1+np.exp(-np.dot(X,w)-b))
    
    # Classify the data
    Y = prob > 0.5
    
    return Y


if __name__ == '__main__':
    np.random.seed(100)
    # Get the training dataset
    X, Y_ = sample_gauss_2d(2, 100)

    # Train the binary logistic regression model
    w, b = binlogreg_train(X, Y_)

    # Evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = probs > 0.5
    
    # Report the performance
    accuracy, recall, precision = eval_perf_binary(Y, Y_)
    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    
    plt.scatter(X[:,0], X[:,1], c=Y_, cmap='viridis')
    plt.show()