import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

""" for now we assume we generate all training data beforehand as numpy arrays.
we then convert to pytorch tensors and store on the cpu memory, converting to
gpu memory when needed. If memory or speed becomes an issue we could rewrite the training data
generation on the gpu using pytorch's linear algebra, and generate it in situ
while training """


def train(
    NN,
    groundtruth_numpy,
    chanvese_numpy,
    epochs=7,
    batch_size=100,
    mu=20,
    lr=0.0001,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
):
    """
    NN is the neural network, e.g. could initialise by NN = SebastianConvNet(1, 256, 256)

    groundtruth_numpy is numpy array of [batchsize, image_channels,
    image_height, image_width] groundtruth segmentations

    chanvese_numpy is numpy array of [batchsize, image_channels, image_height,
    image_width] chanvese segmentations

    The two datasets do not have to contain corresponding images for the purpose
    of training (see paper for why) INFACT THEY SHOULDN'T (should probably
    incorporate this by shuffling beforehand, or could potentially include a
    shuffle command here)

    """

    NN.to(device)
    # not sure why Sebastian doesn't use Adam, but hey
    optimiser = optim.RMSprop(NN.parameters(), lr=lr)

    # object to allow easy access to batches of training data
    dataset = DataLoader(
        TensorDataset(
            torch.from_numpy(groundtruth_numpy).float(),
            torch.from_numpy(chanvese_numpy).float(),
        ),
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True,
    )

    for i in range(epochs):
        """
        haven't got a log keeping track of training progress at the moment
        """
        for groundtruth_batch, chanvese_batch in dataset:
            """
            don't currently do any shuffling of the dataset, just pass through the
            entire dataset (in batches), once per epoch not sure what Sebastian
            does"""
            assert groundtruth_batch.size() == chanvese_batch.size()

            groundtruth_batch = groundtruth_batch.to(device)
            chanvese_batch = chanvese_batch.to(device)

            batchsize = groundtruth_batch.size(0)

            # intermediate point
            epsilon = torch.rand([batchsize], device=device)  # [batchsize]
            epsilon = (
                epsilon.unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .expand(groundtruth_batch.size())
            )  # [batchsize, channels, height, width]
            intermediate_batch = (
                epsilon * groundtruth_batch + (1 - epsilon) * chanvese_batch
            )  # [batchsize, channels, height, width]
            intermediate_batch.requires_grad = True

            # apply the neural network
            groundtruth_NN = NN(groundtruth_batch)  # [batchsize]
            chanvese_NN = NN(chanvese_batch)  # [batchsize]
            intermediate_NN = NN(intermediate_batch)  # [batchsize]
            # [1] trick to compute all gradients in one go
            intermediate_NN = intermediate_NN.sum()

            # calculate the loss
            wasserstein_loss = (groundtruth_NN - chanvese_NN).mean()  # [1]
            # [batchsize, channels, height, width], must create_graph so can
            # backprop again (second derivatives) during gradient descent
            gradient = torch.autograd.grad(
                intermediate_NN, intermediate_batch, create_graph=True
            )[0]
            gradient_loss = (
                (F.relu(gradient.square().sum((1, 2, 3)).sqrt())).square().mean()
            )  # [1]
            loss = wasserstein_loss + mu * gradient_loss  # [1]

            # backprop step
            # no need to zero the gradients of the intermediate point, since it
            # is reinitialised each batch
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

    return NN.to("cpu")
