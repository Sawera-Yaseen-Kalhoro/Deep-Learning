import torch
from torch import nn
import torch.utils
import torch.utils.data
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class PTMNIST(nn.Module):
    def __init__(self):
        super(PTMNIST, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 5, stride = 1, padding =  'same', bias = True)
        # Output = 28x28x16
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # Output = 14x14x16
        self.relu1 = nn.ReLU()
        # Output = 14x14x16
        self.conv2 = nn.Conv2d(16, 32, 5, stride = 1, padding =  'same', bias = True)
        # Output = 14x14x32
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # Output = 7x7x32
        self.relu2 = nn.ReLU()
        # Output = 7x7x32
        self.flatten = nn.Flatten()
        # Output = 7x7x32
        self.fc1 = nn.Linear(32*7*7, 512, bias = True)
        # Output = 512
        self.relu3 = nn.ReLU()
        # Output = 512
        self.logits = nn.Linear(512, 10, bias = True)
        # Output = 10

        self.reset_parameters()

    def reset_parameters(self):
        # This function initializes the weights of convolutional and fully connected layers 
        # using Kaiming Normal initialization to ensure proper gradient flow during backpropagation.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.logits.reset_parameters()

    def forward(self,x):
        x = self.relu1(self.pool1(self.conv1(x)))
        x = self.relu2(self.pool2(self.conv2(x)))
        x = self.relu3(self.fc1(self.flatten(x)))
        x = self.logits(x)
        return x
    
def prepare_data_loaders(batch_size = 64):
    # Load MNIST dataset
    train_data = datasets.MNIST(root = 'Lab1/MNIST', train = True, download = True, transform = ToTensor())
    test_data = datasets.MNIST(root = 'Lab1/MNIST', train = False, download = True, transform = ToTensor())

    # Split train_data into train and validation sets
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) -  train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_data, [train_size, val_size])

    # Create Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

    return train_dataloader, val_dataloader, test_dataloader


def train(model, train_dataloader, val_dataloader, epochs, optimizer, loss_fcn, writer):
    print("Starting the Training")
    
    # Initial evaluation
    train_accuracy, _, _, _, _ = evaluate(model, train_dataloader, loss_fcn)
    print("Initial Accuracy of model:", train_accuracy)
    writer.add_scalar('Accuracy/train', train_accuracy, 0)

    # Initialize lists to store loss values for plotting later
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        
        for i, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad() # clear previous gradients
            outputs = model.forward(images)  # Compute predictions for the batch
            loss = loss_fcn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Log the loss
            running_train_loss += loss.item() 
            writer.add_scalar('Loss/train',loss.item(), epoch * len(train_dataloader) + i)

            if i % 100 == 0:
                print(f'Epoch: {epoch}, Iteration : {i}, Loss: {loss.item()}')

                layer1_weights = model.conv1.weight.data
                grid_image = make_grid(layer1_weights, nrow = 4, normalize = True, scale_each = True)
                writer.add_image(f'conv1_feature_maps_epoch_{epoch}_iteration_{i}', grid_image)

                # Log the accuracy during training
                # accuracy = evaluate(model, train_dataloader)[0].item()
                # print("Train Accuracy:", accuracy)
                # writer.add_scalar('Accuracy/train', accuracy, epoch * len(train_dataloader))

        # Store average training loss for the epoch
        train_losses.append(running_train_loss/ len (train_dataloader))

        # Evaluate training and validation
        # Compute training and validation accuracy and loss after each epoch
        train_accuracy, train_loss, _, train_precision, train_recall = evaluate(model, train_dataloader, loss_fcn)
        val_accuracy, val_loss, _, val_precision, val_recall = evaluate(model, val_dataloader, loss_fcn)

        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)

        print(f"Epoch {epoch} | Train Accuracy: {train_accuracy},Train Loss: {train_loss},Validation Accuracy: {val_accuracy}, Validation Loss: {val_loss}")

        val_losses.append(val_loss)

    plot_loss_evolution(train_losses, val_losses)


def evaluate(model, data_loader, loss_fcn):
    model.eval()
    total_loss = 0.0
    confusion_matrix = torch.zeros(10,10)
    
    # Confusion Matrix
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model.forward(images)
            loss = loss_fcn(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):  # t = true label, p = predicted label
                confusion_matrix[t.long(), p.long()] +=1
    
    # Accuracy
    accuracy = confusion_matrix.diag().sum() / confusion_matrix.sum()
    
    # Average Loss
    avg_loss = total_loss/ len(data_loader)
    
    # Precision and Recall
    precision = confusion_matrix.diag() / confusion_matrix.sum(0) 
    recall = confusion_matrix.diag() / confusion_matrix.sum(1)
    
    return accuracy, avg_loss, confusion_matrix, precision, recall


def plot_loss_evolution(train_losses, val_losses):
    # Plot the loss curves for training and validation
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize = (10,6))
    plt.plot(epochs, train_losses, label = 'Training Loss')
    plt.plot(epochs, val_losses, label = 'Validation Loss')
    plt.title('Loss Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    
    train_dataloader, val_dataloader, test_dataloader = prepare_data_loaders(batch_size = 64)
    
    model = PTMNIST()

    # Regularization to cross-entropy loss
    weight_decay = 0.001
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, weight_decay = weight_decay)

    epochs = 8

    writer = SummaryWriter(comment = f'PTMNIST_lr = 0.001, weight_decay = {weight_decay}')
    train(model, train_dataloader, val_dataloader, epochs, optimizer, loss_fcn, writer)
   
    print("Final Evaluation on Test Data:")
    test_accuracy, test_loss, _, test_precision, test_recall = evaluate(model, test_dataloader, loss_fcn)
    print(f"Test Accuracy: {test_accuracy}, Test Loss: {test_loss}")

    writer.close()


