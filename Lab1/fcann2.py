"""
    Exercise 2: Multi-layer Classification in Python
    A Feed-forward Neural network with one hidden layer trained on 2D dataset.
"""

from data import *
import numpy as np


def softmax(X):
    # Subtract the max value in each row for numerical stability
    max_vals = np.max(X, axis=1, keepdims=True)  # Shape: (N, 1)
    #print('The shape of max_vals is :', max_vals.shape)
    stable_X = X - max_vals

    # Compute the exponentials of the adjusted scores
    expscores = np.exp(stable_X)

    # Normalize by the sum of exponentials in each row
    return expscores / np.sum(expscores, axis=1, keepdims=True)


def fcann2_train(X, Y_, param_niter, param_delta, param_lambda, param_hidden_layer_size):
    """
        - X : Input data 
        - Y_: True class labels
        - param_niter: Number of iterations for training
        - param_delta: Learning rate
        - param_lambda: Regularization parameter
        - param_hidden_layer_size: Number of neurons in the hidden layer
    """
    
    C = int(max(Y_) + 1)  # Number of classes
    N = X.shape[0]        # Number of samples
    

    # Random initialization of weights, Zero initialization of biases
    W1 = np.random.randn(X.shape[1], param_hidden_layer_size) # Input to hidden weights (D x Neurons)
    #print('The shape of W1 is: ',W1.shape)   # 2x5
    
    b1 = np.zeros((1, param_hidden_layer_size)) # Bias vector of hidden layer (1 x Neurons)
    #print('The shape of b1 is : ',b1.shape)  # 1x5
    
    W2 = np.random.randn(param_hidden_layer_size, C) # Hidden to output weights (Neurons x C)
    #print('The shape of W2 is : ',W2.shape) # 5x2
    
    b2 = np.zeros((1, C)) # Bias vector of output layer (1 x C)
    #print('The shape of b2 is : ',b2.shape) # 1x2

    for i in range(param_niter):
        
        # Forward pass

        score1 = np.dot(X, W1) + b1   # Compute scores for hidden layer
        #print('The shape of score1 is :', score1.shape) # 60x5
        
        hidden_layer = np.maximum(0, score1)   # Apply ReLU activation
        #print('The shape of hidden layer is :', hidden_layer.shape) # 60x5
        
        score2 = np.dot(hidden_layer, W2) + b2  # Compute scores for output layer
        #print('The shape of score2 is :', score2.shape) # 60x2
        
        probs = softmax(score2)   # Apply Softmax to get the class probabilities
        #print('The shape of probs is :', probs.shape)  # 60x2

        one_hot_encode = class_to_onehot(Y_)   # Converts Y_ into one-hot encoded matrix
        #print('The shape of one hot encoding is :', one_hot_encode.shape) # 60x2

        # Loss Calculation

        data_loss = -np.sum(one_hot_encode * np.log(probs)) / N  # Negative log likelihood (Measures Prediction Error)
        reg_loss = 0.5 * param_lambda * np.sum(W1 * W1) + 0.5 * param_lambda * np.sum(W2 * W2) # Adds weights penality to avoid overfit
        loss = data_loss + reg_loss

        # Track Loss
        if i % 1000 == 0:
            print("iteration {}: loss {}".format(i, loss))

        # Backpropagation (Gradient Computation)

        gc_dscore2 = probs - one_hot_encode
        grad_W2 = np.dot(hidden_layer.T , gc_dscore2) / N + param_lambda * W2
        grad_b2 = np.sum(gc_dscore2, axis=0, keepdims=True) / N
        gc_dhidden_layer = np.dot(gc_dscore2, W2.T)
        gc_dscore1 = gc_dhidden_layer
        gc_dscore1[score1 <= 0] = 0
        grad_W1 = np.dot(X.T, gc_dscore1) / N + param_lambda * W1
        grad_b1 = np.sum(gc_dscore1, axis=0, keepdims=True) / N

        # Update parameters

        W1 -= param_delta * grad_W1
        b1 -= param_delta * grad_b1
        W2 -= param_delta * grad_W2
        b2 -= param_delta * grad_b2

    return W1, b1, W2, b2

def fcann2_classify(X, W1, b1, W2, b2):
    score1 = np.dot(X, W1) + b1
    hidden_layer = np.maximum(0, score1)  # ReLU Activation
    score2 = np.dot(hidden_layer, W2) + b2
    probs = softmax(score2)   # Softmax Probabilities
    return probs

# def decfun(W1, b1, W2, b2):
#     # Classifies data points based on the trained model.
#     def classify(X):
#       return np.argmax(fcann2_classify(X, W1, b1, W2, b2), axis=1)
#     return classify

def decfun(W1, b1, W2, b2):
    # Classifies data points based on the trained model.
    def classify(X):
      return fcann2_classify(X, W1, b1, W2, b2)[:,1]
    return classify


if __name__ == "__main__":
    np.random.seed(100)

    # Data Generation
    X, Y_ = sample_gmm_2d(6, 2, 10)
    
    # Model Training
    W1, b1, W2, b2 = fcann2_train(X, Y_, param_niter=100000, param_delta=0.05, param_lambda=1e-3, param_hidden_layer_size=5)
    
    # Model Evaluation on the training dataset
    probs = fcann2_classify(X, W1, b1, W2, b2)
    Y = np.argmax(probs, axis=1)
    accuracy, recall, precision = eval_perf_binary(Y, Y_)
    print("Accuracy: {}, recall: {}, precision: {}".format(accuracy, recall, precision))

    # accuracy, pr , M = eval_perf_multi(Y, Y_)
    # print("Accuracy:", accuracy)
    # for i, (recall, precision) in enumerate(pr):
    #     print("Class {}: Recall = {:.2f}, Precision = {:.2f}".format(i, recall, precision))


    # Plot Results
    decfun = decfun(W1, b1, W2, b2)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(decfun, bbox, offset=0.5)
    graph_data(X, Y_, Y)
    plt.show()