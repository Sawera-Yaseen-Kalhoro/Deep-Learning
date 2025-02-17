import os
import random
import torch
import torchvision
from pt_deep import *
from sklearn.svm import SVC

def early_stop_train(model, X, Y, param_niter, param_delta, param_lambda, X_valid, Y_valid):
    flag = True
    for i in range(param_niter):
        train(model, X, Y, 1, param_delta, param_lambda)
        probs = eval(model, X_valid.view(X_valid.shape[0], X_valid.shape[1] * X_valid.shape[2]))
        Y_true = np.argmax(probs, axis = 1)

        accuracy, pr, M = eval_perf_multi(Y_true, Y_valid)

        if i % 100 ==0:
            print(f'Iteration: {i}, loss: {model.loss:.3f}, accuracy: {accuracy:.3f}')

        if model.loss < 0.91 and flag:
            print (f'Stopping early at iteration {i}')
            weights = model.weights
            biases = model.biases
            loss = model.loss
            flag = False
            break

    return weights, biases, loss


def train_mb(model, X, Y, param_niter, param_delta, param_lambda, mini_batches = 100, optimizer = torch.optim.SGD):
    N = X.shape[0]
    stored_loss = []

    for i in range(param_niter):
        shufled_list = torch.randperm(N)
        batch_size = int(N/mini_batches)
        for j in range(mini_batches):
            idx = shufled_list[j * batch_size:(j+1) * batch_size]
            X_mb = X[idx]
            Y_mb = Y[idx]

            train(model, X_mb, Y_mb, 1, param_delta, param_lambda, optimizer)
            stored_loss.append(model.loss.detach().numpy())

        if i % 100 == 0:
            print(f'Iteration: {i}, loss: {model.loss}')

    return stored_loss


def train_linear_svm(X, Y):
    print('Linear SVM Training Started')

    # Checking the shape of input data
    print(f"Shape of X: {X.shape}")
    print(f"Shape of Y: {Y.shape}")

    svm = SVC(kernel = 'linear', decision_function_shape = 'ovo', verbose=True)
    print("SVM model initialized")

    try:
        # Start fitting the model
        print("Fitting the model...")
        svm.fit(X, Y)
        print("Model fitting completed.")
    except Exception as e:
        print(f"Error during fitting: {e}")
        return
    
    # Once the model is fitted, make predictions
    print("Making predictions...")
    Y_predicted = svm.predict(X)
    print(f"Predictions completed. Predicted labels: {Y_predicted[:10]}")  # Show first 10 predictions for quick check

    # Output the results
    accuracy, pr, M = eval_perf_multi(Y_predicted, Y)
    print('Linear SVM')
    print("Accuracy:", accuracy)
    print("Confusion Matrix:", M)
    for i, (recall, precision) in enumerate(pr):
        print(f"Class {i}: Recall = {recall:.2f}, Precision = {precision:.2f}")


def train_kernel_svm(X, Y):
    print('Kernel SVM Training Started')

    # Checking the shape of input data
    print(f"Shape of X: {X.shape}")
    print(f"Shape of Y: {Y.shape}")

    # Initialize the SVM model with RBF kernel
    svm = SVC(kernel='rbf', decision_function_shape='ovo', verbose=True)
    print("SVM model initialized")

    try:
        # Start fitting the model
        print("Fitting the model...")
        svm.fit(X, Y)
        print("Model fitting completed.")
    except Exception as e:
        print(f"Error during fitting: {e}")
        return

    # Once the model is fitted, make predictions
    print("Making predictions...")
    Y_predicted = svm.predict(X)
    print(f"Predictions completed. Predicted labels: {Y_predicted[:10]}")  # Show first 10 predictions for quick check

    # Evaluate the performance
    print("Evaluating performance...")
    accuracy, pr, M = eval_perf_multi(Y_predicted, Y)

    # Output the results
    print('Kernel SVM Results:')
    print("Accuracy:", accuracy)
    print("Confusion Matrix:", M)
    for i, (recall, precision) in enumerate(pr):
        print(f"Class {i}: Recall = {recall:.2f}, Precision = {precision:.2f}")
    


if __name__ == '__main__':
    file_dirct = os.path.dirname(os.path.abspath(__file__))

    dataset_root = 'MNIST'  # change this to your preference
    mnist_train = torchvision.datasets.MNIST(dataset_root, train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(dataset_root, train=False, download=True)

    x_train, y_train = mnist_train.data, mnist_train.targets
    x_test, y_test = mnist_test.data, mnist_test.targets
    x_train, x_test = x_train.float().div_(255.0), x_test.float().div_(255.0)

    # Visualize random images from the training set
    # random_integers = [random.randint(0, 30000) for _ in range(200)]
    # for i in random_integers:
    #     plt.imshow(x_train[i].numpy(), cmap = 'gray')
    #     plt.show()


    N = x_train.shape[0]
    D = x_train.shape[1] * x_train.shape[2]
    C = y_train.max().add_(1).item()

    y_train_oh = class_to_onehot(y_train)
    N_valid = int(N/5)
    shuffled_data_list = torch.randperm(N)

    x_valid = x_train[shuffled_data_list[:N_valid]]
    y_valid = y_train[shuffled_data_list[:N_valid]]


    ptlr = PTDeep([784,10], torch.relu)


    """
        Train and explain the loss function of a randomly initialized model.
    """
       
    print("Analyzing loss for a randomly initialized model with random data...")

    config = [784, 10]
    model = PTDeep(config)

    # Generate random input data and labels
    random_x = np.random.rand(100, 784)  # 100 random input samples with 784 features
    random_y = torch.randint(0, 10, (100,))  # Random labels for 10 classes

    # Define the loss function
    nll_loss = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        # Pass the random data through the model
        logits = eval(model, random_x)
        logits = torch.Tensor(logits)  # Convert to Tensor
        
        # Compute loss
        loss = nll_loss(logits, random_y)
        print(f"Initial loss = {loss}")


    """
    Training Functions
    """

    # Train with full batch
    print('Training with full batch')
    stored_loss = train(ptlr, x_train.view(N, D), torch.Tensor(y_train_oh), 10000, 0.2, 0.01)
    plt.plot(stored_loss)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    # Plot the images which contribute the most to the loss
    image_indices = [indices[1] for indices in ptlr.most_loss_images]
    for i in image_indices:
        plt.imshow(x_train[i].numpy(), cmap = 'gray')
        plt.show()

    # Train with early stopping
    #weights, biases, loss = early_stop_train(ptlr, x_train.view(N, D), torch.Tensor(y_train_oh), 8000, 0.2, 0.1, x_valid, y_valid)

    # Train with mini-batches
    # print('Training with mini-batches')
    # stored_loss = train_mb(ptlr, x_train.view(N, D), torch.Tensor(y_train_oh), 2000, 0.1, 1e-4, 100, optimizer = optim.SGD)
    # plt.plot(stored_loss)
    # plt.show()

    # Train using linear SVM
    #train_linear_svm(x_train.view(N,D).numpy(), y_train.numpy())

    # Train using kernel SVM
    #train_kernel_svm(x_train.view(N,D).numpy(), y_train.numpy())

    #Gnerate the compuatation graph
    #make_dot(ptlr.prob, params = ptlr.state_dict()).render("MNISTgraph", directory = file_dirct, format = 'png', cleanup = True)

    # weights = ptlr.weights
    # for i, w in enumerate(weights):
    #     for i in range(w.size(1)):
    #         weight = w[:, i].detach().view(28,28).numpy()
    #         weight = (((weight - weight.min())/ (weight.max()- weight.min())) * 255.0).astype(np.uint8)
    #         plt.imshow(weight, cmap = 'viridis')
    #         plt.title('Weights for class {}'. format(i))
    #         plt.show()

    #     torch.save(ptlr.state_dict(), 'saved_weights/-model_weights.pth')

    #print("Total number of parameters:", count_params(ptlr))

    #Print Evaluation metrics for training set
    print("Training set metrics")
    probs = eval(ptlr, x_train.view(N, D))
    Y = np.argmax(probs , axis = 1)
    accuracy, pr, M = eval_perf_multi(Y, y_train)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:", M)
    for i, (recall, precision) in enumerate(pr):
        print("Class {}: Recall = {:.2f}, Precision = {:.2f}".format(i, recall, precision))


    # Print evaluation metrics for the  test set
    print("Test set metrics")
    N_test = x_test.shape[0]
    D_test = x_test.shape[1] * x_test.shape[2]
    probs = eval(ptlr, x_test.view(N_test, D_test))
    probs = eval(ptlr, x_test.view(N_test, D_test))
    Y = np.argmax(probs, axis = 1)
    accuracy, pr, M = eval_perf_multi(Y, y_test)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:", M)
    for i, (recall, precision) in enumerate(pr):
        print("Class {}: Recall = {:.2f}, Precision = {:.2f}".format(i, recall, precision))


    # Print evaluation metrics for the validation set
    # print('Validation set Metrics')
    # N_valid = x_valid.shape[0]
    # D_valid = x_valid.shape[1]*x_valid.shape[2]
    # probs = eval(ptlr, x_valid.view(N_valid, D_valid))
    # Y = np.argmax(probs, axis=1)
    # accuracy, pr, M = eval_perf_multi(Y, y_valid)
    # print("Accuracy:", accuracy)
    # print("Confusion Matrix:", M)
    # for i, (recall, precision) in enumerate(pr):
    #     print("Class {}: Recall = {:.2f}, Precision = {:.2f}".format(i, recall, precision))


    # Print evaluation metrics for the early stopping
    # ptlr.weights = weights
    # ptlr.biases = biases
    # print('Early stopping Metrics')
    # probs = eval(ptlr, x_test.view(N_test, D_test))
    # Y = np.argmax(probs, axis=1)
    # # print out the performance metric (precision and recall per class)
    # accuracy, pr, M = eval_perf_multi(Y, y_test)
    # print("Accuracy:", accuracy)
    # print("Confusion Matrix:", M)
    # for i, (recall, precision) in enumerate(pr):
    #     print("Class {}: Recall = {:.2f}, Precision = {:.2f}".format(i, recall, precision))
    