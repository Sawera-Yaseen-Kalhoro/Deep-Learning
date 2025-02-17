import torch
import torch.nn as nn
import torch.nn.functional as F


class _BNReluConv(nn.Sequential):
    def __init__(self, num_maps_in, num_maps_out, k=3, bias=True):
        super(_BNReluConv, self).__init__()
        # YOUR CODE HERE
        self.append(nn.GroupNorm(1, num_maps_in))
        self.append(nn.ReLU())
        self.append(nn.Conv2d(num_maps_in, num_maps_out, kernel_size = k, bias = bias))


class SimpleMetricEmbedding(nn.Module):
    def __init__(self, input_channels, emb_size=32):
        super().__init__()
        self.emb_size = emb_size
        # YOUR CODE HERE
        self.conv1 = _BNReluConv(input_channels, self.emb_size)
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv2 = _BNReluConv(self.emb_size, self.emb_size)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv3 = _BNReluConv(self.emb_size, self.emb_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def get_features(self, img):
        # Returns tensor with dimensions BATCH_SIZE, EMB_SIZE
        # YOUR CODE HERE
        x = self.conv1(img)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        return x

    def loss(self, anchor, positive, negative, margin = 1.0):
        a_x = self.get_features(anchor)
        p_x = self.get_features(positive)
        n_x = self.get_features(negative)
        # YOUR CODE HERE
        loss = F.relu(F.pairwise_distance(a_x, p_x, p = 2) - F.pairwise_distance(a_x, n_x, p = 2) + margin).mean()
        return loss

