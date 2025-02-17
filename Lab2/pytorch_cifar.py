import os
import pickle
import math
import numpy as np
import skimage as ski
import skimage.io
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import ToTensor
from torchvision import datasets
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.tensorboard import SummaryWriter


class PTCIFAR(nn.Module):
    def __init__(self):
        super(PTCIFAR, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 5, stride = 1, padding = 'same', bias = True)
        # out = 16x32x32
        self.relu1 = nn.ReLU()
        # out = 16x32x32
        self.pool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        # out = 16x16x16
        self.conv2 = nn.Conv2d(16, 32, 5, stride = 1, padding = 'same', bias = True)
        #out = 16x16x16
        self.relu2 = nn.ReLU()
        #out = 16x16x16
        self.pool2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        # out = 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32*7*7, 256, bias = True)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128, bias = True)
        self.relu4 = nn.ReLU()
        self.logits = nn.Linear(128, 10, bias = True)

        self.reset_parameters

    def reset_parameters(self):
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
        x = self.relu4(self.fc2(x))
        x = self.logits(x)
        
        return x
    

def plot_training_progress(save_dir, data):
  fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16,8))

  linewidth = 2
  legend_size = 10
  train_color = 'm'
  val_color = 'c'

  num_points = len(data['train_loss'])
  x_data = np.linspace(1, num_points, num_points)
  ax1.set_title('Cross-entropy loss')
  ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax1.legend(loc='upper right', fontsize=legend_size)
  ax2.set_title('Average class accuracy')
  ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='train')
  ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
           linewidth=linewidth, linestyle='-', label='validation')
  ax2.legend(loc='upper left', fontsize=legend_size)
  ax3.set_title('Learning rate')
  ax3.plot(x_data, data['lr'], marker='o', color=train_color,
           linewidth=linewidth, linestyle='-', label='learning_rate')
  ax3.legend(loc='upper left', fontsize=legend_size)

  save_path = os.path.join(save_dir, 'training_plot.png')
  print('Plotting in: ', save_path)
  plt.savefig(save_path)
  plt.show()


def train(model, train_dataloader, valid_dataloader, epochs, optimizer, loss_fcn, writer, scheduler):
    plot_data = {'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': [], 'lr': []}
    print("Startting Training")
    initial_train_accuracy, _, _, _, _ = evaluate(model, train_dataloader)
    print("Initial Accuracy of model:", initial_train_accuracy)
    initial_valid_accuracy, _, _, _, _ = evaluate(model, valid_dataloader)

    initial_accuracy_dict = {'train': initial_train_accuracy, 'validation': initial_valid_accuracy}
    writer.add_scalars('Accuracy', initial_accuracy_dict)

    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad() # clear previous gradients
            outputs = model.forward(images)  # Compute predictions for the batch
            loss = loss_fcn(outputs, labels)
            loss.backward()
            optimizer.step()

            # Log the loss
            # writer.add_scalar('Loss/train',loss.item(), epoch * len(train_dataloader) + i)

            if i % 100 == 0:
                print(f'Epoch: {epoch}, Iteration : {i}, Loss: {loss.item()}')

                layer1_weights = model.conv1.weight.data
                grid_image = make_grid(layer1_weights, nrow = 4, normalize = True, scale_each = True)
                writer.add_image(f'conv1_feature_maps_epoch_{epoch}_iteration_{i}', grid_image)

        # Step the scheduler
        scheduler.step()

        # Evaluate training and validation
        train_accuracy, train_loss, _, train_precision, train_recall = evaluate(model, train_dataloader)
        valid_accuracy, valid_loss, _, valid_precision, valid_recall = evaluate(model, valid_dataloader)


        # Append metrics to plot_data
        plot_data['train_loss'].append(train_loss)
        plot_data['valid_loss'].append(valid_loss)
        plot_data['train_acc'].append(train_accuracy.item())
        plot_data['valid_acc'].append(valid_accuracy.item())
        plot_data['lr'].append(optimizer.param_groups[0]['lr'])
        
        print('Train Accuracy: ', train_accuracy.item())
        loss_dict = {'train': train_loss, 'validation': valid_loss}
        accuracy_dict = {'train': train_accuracy.item(), 'validation': valid_accuracy.item()}

        writer.add_scalars('Loss', loss_dict, epoch*len(train_dataloader))
        writer.add_scalars('Accuracy', accuracy_dict, epoch*len(train_dataloader))
        
    print('Finished Training')
    return plot_data


def evaluate(model, data_loader):
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

def draw_image(img, mean, std):
  img = img.permute(1, 2, 0)
  img *= std
  img += mean
  img = img.cpu().numpy().astype(np.uint8)
  ski.io.imshow(img)
  #ski.io.show()

def plot_loss_images(model, training_data, mean, std):
    loss_fcn = nn.CrossEntropyLoss(reduction = 'none')   #loss is computed per sample not averaged across batch
    train_dataloader = DataLoader(training_data, batch_size = 40000)
    images, labels = next(iter(train_dataloader))
    outputs = model.forward(images)
    loss = loss_fcn(outputs, labels)
    _, predicted = torch.max(outputs.data, 1)
    
    # Plot the 20 incorrectly classified images with the largest loss
    
    incorrect_images = images[predicted != labels]
    incorrect_labels = labels[predicted != labels]
    incorrect_predicted = predicted[predicted != labels]
    incorrect_loss = loss[predicted != labels]

    # Get the top 3 predicted classes for each image
    _, top3_indices = torch.topk(outputs, 3, dim=1)

    # Filter top-3 predictions for only incorrect images
    incorrect_top3_indices = top3_indices[predicted != labels]

    # Sort the incorrect losses and apply the same indexing to incorrect_top3_indices
    _, indices = torch.sort(incorrect_loss, descending=True)

    # Select the top 20 images with the largest loss
    incorrect_images = incorrect_images[indices[:20]]
    incorrect_labels = incorrect_labels[indices[:20]]
    incorrect_predicted = incorrect_predicted[indices[:20]]
    top3_classes = incorrect_top3_indices[indices[:20]]

    # Plot the images and label with the true and predicted label
    
    # Convert torch format (batch, channels, height, width) to matplotlib format (batch, height, width, channels).
    # incorrect_images = incorrect_images.permute(0, 2, 3, 1)  
    # incorrect_images = incorrect_images.numpy()
    # incorrect_labels = incorrect_labels.numpy()
    # incorrect_predicted = incorrect_predicted.numpy()

    figure = plt.figure(figsize=(8,8))
    for i in range(20):
        ax = figure.add_subplot(5, 4, i+1)
        #ax.imshow(incorrect_images[i])
        draw_image(incorrect_images[i], mean, std)
        ax.set_title(f"True: {incorrect_labels[i]}\nPred: {incorrect_predicted[i]}\nTop 3: {top3_classes[i].numpy()}")
        #ax.set_title(f'True: {incorrect_labels[i]}, predicted: {incorrect_predicted[i]}')
        ax.axis('off')
    plt.show()


def multiclass_hinge_loss(logits: torch.Tensor, target: torch.Tensor, delta = 1.):
    """
        Args:
            logits: torch.Tensor with shape (B,C), 
            where B is batch size, and C is number of classes

            target: torch.LongTensor with shape (B, ) representing ground truth labels.
            delta: Hyperparameter
            Returns: Loss as scalar torch.Tensor
    """

    one_hot_label = torch.nn.functional.one_hot(target, num_classes)
    mask = 1 - one_hot_label

    correct_logits = torch.diag(torch.sum(one_hot_label * logits, dim = 1))
    correct_logits_matrix = torch.mm(correct_logits, torch.ones_like(logits))

    errors = torch.relu(logits - correct_logits_matrix + delta)

    sample_losses = torch.sum(errors * mask, dim = 1)

    loss = torch.mean(sample_losses)

    return loss


def shuffle_data(data_x, data_y):
  indices = np.arange(data_x.shape[0])
  np.random.shuffle(indices)
  shuffled_data_x = np.ascontiguousarray(data_x[indices])
  shuffled_data_y = np.ascontiguousarray(data_y[indices])
  return shuffled_data_x, shuffled_data_y

def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict


if __name__ == "__main__":

    DATA_DIR = 'Lab 2/cifar-10-python/cifar-10-batches-py'
    SAVE_DIR = 'Lab 2/training_plot'

    # Create the directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)


    img_height = 32
    img_width = 32
    num_channels = 3
    num_classes = 10

    train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
        subset = unpickle(os.path.join(DATA_DIR, 'data_batch_%d' % i))
        train_x = np.vstack((train_x, subset['data']))
        train_y += subset['labels']
    train_x = train_x.reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1)
    train_y = np.array(train_y, dtype=np.int32)

    subset = unpickle(os.path.join(DATA_DIR, 'test_batch'))
    test_x = subset['data'].reshape((-1, num_channels, img_height, img_width)).transpose(0, 2, 3, 1).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)

    valid_size = 5000
    train_x, train_y = shuffle_data(train_x, train_y)
    valid_x = train_x[:valid_size, ...]
    valid_y = train_y[:valid_size, ...]
    train_x = train_x[valid_size:, ...]
    train_y = train_y[valid_size:, ...]
    
    data_mean = train_x.mean((0, 1, 2))
    data_std = train_x.std((0, 1, 2))

    train_x = (train_x - data_mean) / data_std
    valid_x = (valid_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std

    train_x = train_x.transpose(0, 3, 1, 2)
    valid_x = valid_x.transpose(0, 3, 1, 2)
    test_x = test_x.transpose(0, 3, 1, 2)


    train_x = torch.from_numpy(train_x)
    valid_x = torch.from_numpy(valid_x)
    test_x = torch.from_numpy(test_x)
    train_y = (torch.from_numpy(train_y)).long()
    valid_y = (torch.from_numpy(valid_y)).long()
    test_y = (torch.from_numpy(test_y)).long()

    training_data = TensorDataset(train_x, train_y)
    validation_data = TensorDataset(valid_x, valid_y)
    test_data = TensorDataset(test_x, test_y)
    training_data = datasets.CIFAR10(root="Lab2/cifar10",
                                      train=True, download=True, transform=ToTensor())
    

    # Define data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    training_data = datasets.CIFAR10(root="Lab2/cifar10",
                                      train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root="Lab2/cifar10",
                                    train=False, download=True, transform=transform)
    
    
    # separate 5000 examples from the training set to create a validation set randomly
    training_data, validation_data = torch.utils.data.random_split(training_data, [0.9, 0.1])
    
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(validation_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data)


    lr = 0.01   
    weight_decay = 0.001
    epochs = 20   
    step_size = 2  # Reduce LR every 2 epochs  
    gamma = 0.1  # Multiply LR by this factor at each step 

    model = PTCIFAR()
    loss_fcn = nn.CrossEntropyLoss()
    #loss_fcn = multiclass_hinge_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size = step_size, gamma = gamma)
    writer = SummaryWriter(comment=f'PTCIFAR_lr={lr},weight_decay={weight_decay}')
    plot_data = train(model, train_dataloader, valid_dataloader, epochs, optimizer, loss_fcn, writer, scheduler)
    writer.close()

    plot_training_progress(SAVE_DIR, plot_data)
    
    accuracy,loss, confusion_matrix, precision, recall = evaluate(model, test_dataloader)
    print(f'Test Accuracy: {accuracy}')
    print(f'Confusion Matrix: {confusion_matrix}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

    plot_loss_images(model, training_data, data_mean, data_std)
    

