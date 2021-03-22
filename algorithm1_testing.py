import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ClassFiles.DataLoader import get_generated_dataloader

"""
for now we assume we generate all training data beforehand as numpy arrays.
we then convert to pytorch tensors and store on the cpu memory, converting to
gpu memory when needed. If memory or speed becomes an issue we could rewrite the
training data generation on the gpu using pytorch's linear algebra, and generate
it in situ while training
"""



def train(
    NN,
    SAVE_PATH = None,
    epochs=100,
    batch_size=20,
    mu=1,
    shuffle = True,
    binary = False,
    flip = False,
    l2_gradients = False,
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
    
    Binary converts the chan-vese segmentations to binary images
    
    Flip randomly flips white and black images

    """

    NN.to(device)

    optimiser = optim.Adam(NN.parameters())
    
    
    groundtruth_loader = get_generated_dataloader('train', 'clean', batch_size, shuffle = shuffle)
    chanvese_loader = get_generated_dataloader('train', 'chan-vese', batch_size, shuffle = shuffle)
    
    
    
    """
    the next few lines are to evaluate performance before training has started, to get a control
    """
    eval_groundtruth_loader = get_generated_dataloader('eval', 'clean', batch_size = 100)
    eval_chanvese_loader = get_generated_dataloader('eval', 'chan-vese', batch_size = 100)
    
    
    eval_groundtruth_batch = iter(eval_groundtruth_loader).next()[0].to(device)
    eval_chanvese_batch = iter(eval_chanvese_loader).next()[0].to(device)
    
    if binary == True:
        """
        make chanvese image binary, and then subtract 0.5
        """
        eval_chanvese_batch = (eval_chanvese_batch > 0.5).float()
    
    """
    treat white and black equally (back is 0.5, white is - 0.5)
    """
    eval_groundtruth_batch = eval_groundtruth_batch - 0.5
    eval_chanvese_batch = eval_chanvese_batch - 0.5
    
    with torch.no_grad():
        groundtruth_value = NN(eval_groundtruth_batch)
        chanvese_value = NN(eval_chanvese_batch)
    """
    print the average 'distance' the neural network has learnt between groundtruth and chanvese images
    also print the max value on groundtruth, and the min on chanvese,
    this gives an indication of its performance as a classifier
    """
    print('untrained performance', (groundtruth_value.mean() - chanvese_value.mean()).item(), groundtruth_value.view(-1).max(0)[0].item(), chanvese_value.view(-1).min(0)[0].item(), (groundtruth_value - chanvese_value).view(-1).max(0)[0].item(), ((groundtruth_value - chanvese_value) > 0).float().mean().item())
    
    for i in range(epochs):
        
        assert len(groundtruth_loader) == len(chanvese_loader)
        
        groundtruth_iter = iter(groundtruth_loader)
        chanvese_iter = iter(chanvese_loader)
        
        for i in range(len(groundtruth_loader)):
            groundtruth_batch = groundtruth_iter.next()[0]
            chanvese_batch = chanvese_iter.next()[0]
            
            assert groundtruth_batch.size() == chanvese_batch.size()
            batchsize = groundtruth_batch.size(0)
            
            groundtruth_batch = groundtruth_batch.to(device)
            chanvese_batch = chanvese_batch.to(device)
            
            if binary == True:
                """
                make chanvese image binary, and then subtract 0.5
                """
                chanvese_batch = (chanvese_batch > 0.5).float()
            
            """
            treat white and black equally (back is 0.5, white is - 0.5)
            """
            groundtruth_batch = groundtruth_batch - 0.5
            chanvese_batch = chanvese_batch - 0.5
            
            if flip == True:
                """
                force it to treat white as black and vica versa
                (randomly flip colours, and flip independently for groundtruth and chanvese)
                """
                flip_values = ((torch.rand([batchsize], device = device) > 0.5) * 2 - 1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                groundtruth_batch = flip_values * groundtruth_batch
                chanvese_batch = flip_values * chanvese_batch
            
            # REVIEW: Unsqueezing over the 1-axis is enough for batchwise multiplication
            epsilon = torch.rand([batchsize], device=device).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            
            intermediate_batch = (
                epsilon * groundtruth_batch + (1 - epsilon) * chanvese_batch
            )  # [batchsize, channels, height, width]
            intermediate_batch.requires_grad = True
            assert intermediate_batch.isnan().sum() == 0

            # apply the neural network
            groundtruth_out = NN(groundtruth_batch)  # [batchsize]
            chanvese_out = NN(chanvese_batch)  # [batchsize]
            intermediate_out = NN(intermediate_batch)  # [batchsize]

            # REVIEW: Why mean() and not sum()?
            # calculate the loss
            wasserstein_loss = (groundtruth_out - chanvese_out).mean()  # [1]
            assert wasserstein_loss.isnan().sum() == 0

            # Set 'create_graph=True' so we can backprop a function of the
            # gradient (--> second derivatives). This is needed for implementing
            # the approximate 1-Lipschitz constraint.
            # REVIEW: Maybe we should use "retain_graph"?
            gradient = torch.autograd.grad(
                intermediate_out.sum(), intermediate_batch, create_graph=True
            )[0]
            # --> [batchsize, channels, height, width]
            assert gradient.isnan().sum() == 0
            
            """
            instead of making the neural network lipschitz in an L^2 sense
            (where in an L^2 sense refers to how we calculate the 'size' of the derivative),
            make each partial derivative lipschitz
            (and it is quite easy to define what the size of each partial derivative is)
            
            ends up enforcing that the 'wasserstein_loss' actually corresponds to how many pixels it thinks are different
            """
            
            if l2_gradients == False:
                gradient_loss = (
                    (F.relu(gradient.abs() - 1).sum((1, 2, 3))).mean() #mean is only over batch dimension
                )  # [1]
            else:
                gradient_loss = (
                (F.relu(gradient.square().sum((1, 2, 3)).sqrt() - 1)).square().mean()
            )  # [1]
            
            """
            mu = 1,
            since wasserstein_loss is scaled such that each pixel corresponds to an O(1) change in output, and
            gradient loss is similarly scaled above
            """
            loss = (wasserstein_loss + mu * gradient_loss)  # [1]
            assert loss.isnan().sum() == 0
            print(wasserstein_loss.item(), gradient_loss.item())

            # backprop step
            # no need to zero the gradients of the intermediate point, since it
            # is reinitialised each batch
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            assert NN.conv1.weight.isnan().sum() == 0
        
        
        """
        the next few lines are for evaluating the performance of the NN at the end of this epoch
        """
        eval_groundtruth_batch = iter(eval_groundtruth_loader).next()[0].to(device)
        eval_chanvese_batch = iter(eval_chanvese_loader).next()[0].to(device)
        
        if binary == True:
            """
            make chanvese image binary, and then subtract 0.5
            """
            eval_chanvese_batch = (eval_chanvese_batch > 0.5).float()

        """
        treat white and black equally (back is 0.5, white is - 0.5)
        """
        eval_groundtruth_batch = eval_groundtruth_batch - 0.5
        eval_chanvese_batch = eval_chanvese_batch - 0.5
        
        with torch.no_grad():
            groundtruth_value = NN(eval_groundtruth_batch)
            chanvese_value = NN(eval_chanvese_batch)
        
        """
        print the average 'distance' the neural network has learnt between groundtruth and chanvese images
        also print the max value on groundtruth, and the min on chanvese,
        this gives an indication of its performance as a classifier
        """
        print('done epoch', (groundtruth_value.mean() - chanvese_value.mean()).item(), groundtruth_value.view(-1).max(0)[0].item(), chanvese_value.view(-1).min(0)[0].item(), (groundtruth_value - chanvese_value).view(-1).max(0)[0].item(), ((groundtruth_value - chanvese_value) > 0).float().mean().item())
    
    if SAVE_PATH != None:
        torch.save(NN.state_dict(), SAVE_PATH)

    return NN.to("cpu")
