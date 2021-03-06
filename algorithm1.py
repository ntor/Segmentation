import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ClassFiles.DataLoader import get_generated_dataloader

""" for now we assume we generate all training data beforehand as numpy arrays.
we then convert to pytorch tensors and store on the cpu memory, converting to
gpu memory when needed. If memory or speed becomes an issue we could rewrite the
training data generation on the gpu using pytorch's linear algebra, and generate
it in situ while training """



def train(
    NN, SAVE_PATH = None
    epochs=100,
    batch_size=20,
    mu=20,
    lr=0.0001,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
):
    """
    NN is the neural network, e.g. could initialise by NN = SebastianConvNet(1, 256, 256)
    
    SAVE_PATH is a string, where to save the parameters of the NN after training is complete (if None then doesn't save)

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
    
    # REVIEW: Just one dataloader (for Lucas' pleasure)
    groundtruth_loader = get_generated_dataloader('train', 'clean', batch_size)
    chanvese_loader = get_generated_dataloader('train', 'chan-vese', batch_size)
    
    eval_groundtruth_loader = get_generated_dataloader('eval', 'clean', batch_size=100, shuffle=False)
    eval_chanvese_loader = get_generated_dataloader('eval', 'chan-vese', batch_size=100, shuffle=False)
    
    eval_groundtruth_batch = iter(eval_groundtruth_loader).next()[0].to(device)
    eval_chanvese_batch = iter(eval_chanvese_loader).next()[0].to(device)

    with torch.no_grad():
        groundtruth_mean_value = NN(eval_groundtruth_batch).mean().item()
        chanvese_mean_value = NN(eval_chanvese_batch).mean().item()
    print('untrained performance', groundtruth_mean_value, chanvese_mean_value)
    
    for i in range(epochs):
        """
        haven't got a log keeping track of training progress at the moment
        """
        
        
        assert len(groundtruth_loader) == len(chanvese_loader)
        
        groundtruth_iter = iter(groundtruth_loader)
        chanvese_iter = iter(chanvese_loader)
        
        for i in range(len(groundtruth_loader)):
            groundtruth_batch = groundtruth_iter.next()[0]
            chanvese_batch = chanvese_iter.next()[0]
            
            assert groundtruth_batch.size() == chanvese_batch.size()

            groundtruth_batch = groundtruth_batch.to(device)
            chanvese_batch = chanvese_batch.to(device)

            batchsize = groundtruth_batch.size(0)

            # REVIEW: Unsqueezing over the 1-axis is enough for batchwise multiplication
            epsilon = torch.rand([batchsize], device=device).unsqueeze(1).unsqueeze(2).unsqueeze(3)

            intermediate_batch = (
                epsilon * groundtruth_batch + (1 - epsilon) * chanvese_batch
            )  # [batchsize, channels, height, width]
            intermediate_batch.requires_grad = True

            # apply the neural network
            groundtruth_out = NN(groundtruth_batch)  # [batchsize]
            chanvese_out = NN(chanvese_batch)  # [batchsize]
            intermediate_out = NN(intermediate_batch)  # [batchsize]

            # REVIEW: Why mean() and not sum()?
            # calculate the loss
            wasserstein_loss = (groundtruth_out - chanvese_out).mean()  # [1]

            # Set 'create_graph=True' so we can backprop a function of the
            # gradient (--> second derivatives). This is needed for implementing
            # the approximate 1-Lipschitz constraint.
            # REVIEW: Maybe we should use "retain_graph"?
            gradient = torch.autograd.grad(
                intermediate_out.sum(), intermediate_batch, create_graph=True
            )[0]
            # --> [batchsize, channels, height, width]

            gradient_loss = (
                (F.relu(gradient.square().sum((1, 2, 3)).sqrt() - 1)).square().mean()
            )  # [1]
            loss = wasserstein_loss + mu * gradient_loss  # [1]

            # backprop step
            # no need to zero the gradients of the intermediate point, since it
            # is reinitialised each batch
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        
        eval_groundtruth_batch = iter(eval_groundtruth_loader).next()[0].to(device)
        eval_chanvese_batch = iter(eval_chanvese_loader).next()[0].to(device)
        with torch.no_grad():
            groundtruth_mean_value = NN(eval_groundtruth_batch).mean().item()
            chanvese_mean_value = NN(eval_chanvese_batch).mean().item()
        print('done epoch', groundtruth_mean_value, chanvese_mean_value)
    
    if SAVE_PATH != None:
        torch.save(NN.state_dict(), SAVE_PATH)

    return NN.to("cpu")
