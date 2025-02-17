import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import defaultdict
from random import choice



class MNISTMetricDataset(Dataset):
    def __init__(self, root="Lab1/MNIST", split='train', remove_class = None):
        super().__init__()
        assert split in ['train', 'test', 'traineval']
        self.root = root
        self.split = split
        mnist_ds = torchvision.datasets.MNIST(self.root, train='train' in split, download=True)
        self.images, self.targets = mnist_ds.data.float() / 255., mnist_ds.targets
        self.classes = list(range(10))

        if remove_class is not None:
            # Get indices of all images that belong to the class to be removed
            indices_to_remove = [i for i in range(len(self.targets)) if self.targets[i].item() == remove_class]
            # remove images and targets
            mask = torch.ones_like(self.targets, dtype = torch.bool)
            mask[indices_to_remove] = 0
            self.targets = self.targets[mask]
            self.images = self.images[mask]

        self.target2indices = defaultdict(list)
        for i in range(len(self.images)):
            self.target2indices[self.targets[i].item()] += [i]

    def _sample_negative(self, index):
        # YOUR CODE HERE
        positive_class = self.targets[index].item()
        negative_classes = self.target2indices.copy()
        del negative_classes[positive_class]
        random_negative_class = choice(list(negative_classes.keys()))
        d = choice(negative_classes[random_negative_class])
        return d


    def _sample_positive(self, index):
        # YOUR CODE HERE
        positive_class = self.targets[index].item()
        positive_targets = self.target2indices[positive_class]
        c = choice(positive_targets)
        return c


    def __getitem__(self, index):
        anchor = self.images[index].unsqueeze(0)
        target_id = self.targets[index].item()
        if self.split in ['traineval', 'val', 'test']:
            return anchor, target_id
        else:
            positive = self._sample_positive(index)
            negative = self._sample_negative(index)
            positive = self.images[positive]
            negative = self.images[negative]
            return anchor, positive.unsqueeze(0), negative.unsqueeze(0), target_id

    def __len__(self):
        return len(self.images)
    

if __name__ == '__main__':
    #dataset = MNISTMetricDataset
    dataset = MNISTMetricDataset(remove_class = 0)
    while True:
        random_idx = choice(range(len(dataset)))
        anchor, positive, negative, target_id = dataset[random_idx]
        # Use matplotlib to show the images
        fig = plt.figure()
        ax1 = fig.add_subplot(1,3,1)
        ax1.imshow(anchor.squeeze(0), cmap = 'gray')

        ax1 = fig.add_subplot(1,3,2)
        ax1.imshow(positive.squeeze(0), cmap = 'gray')

        ax1 = fig.add_subplot(1,3,3)
        ax1.imshow(negative.squeeze(0), cmap = 'gray')

        plt.show()
