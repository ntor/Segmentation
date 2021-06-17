import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train_regulariser(
    NN,
    good_dataloader,
    bad_dataloader,
    epochs=20,
    lr=0.001,
    mu=1,
    binary=True,
    flip=False,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    save_path=None,
):
    """Train a neural network 'NN' to be a distance function between 'good'
    data (provided by 'good_dataloader') and "bad" data (provided by
    'bad_dataloader'), such that it attains high values on bad, and low values
    on good data, respectively.

    Parameters:

    'NN': neural network, e.g. could initialised by NN = SebastianConvNet(1, 256,
    256)

    'binary': converts the chan-vese segmentations to binary images

    'flip': randomly flips values of segmentation u to (1-u)

    """

    NN.to(device)
    optimiser = optim.Adam(NN.parameters(), lr=lr)

    for j in range(epochs):
        assert len(good_dataloader) == len(bad_dataloader)
        good_iter = iter(good_dataloader)
        bad_iter = iter(bad_dataloader)

        for i in range(len(good_dataloader)):
            groundtruth_batch = good_iter.next()
            chanvese_batch = bad_iter.next()

            assert groundtruth_batch.size() == chanvese_batch.size()
            batchsize = groundtruth_batch.size(0)

            groundtruth_batch = groundtruth_batch.to(device)
            chanvese_batch = chanvese_batch.to(device)

            if binary:
                # make chanvese image binary, and then subtract 0.5
                chanvese_batch = (chanvese_batch > 0.5).float()

            # treat white and black equally (back is 0.5, white is - 0.5)
            groundtruth_batch -= 0.5
            chanvese_batch -= 0.5

            if flip:
                # force it to treat white as black and vica versa
                # (randomly flip colours, and flip independently for groundtruth and chanvese)
                flip_values = (
                    ((torch.rand([batchsize], device=device) > 0.5) * 2 - 1)
                    .unsqueeze(1)
                    .unsqueeze(2)
                    .unsqueeze(3)
                )
                groundtruth_batch = flip_values * groundtruth_batch
                chanvese_batch = flip_values * chanvese_batch

            epsilon = (
                torch.rand([batchsize], device=device)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
            )

            intermediate_batch = (
                epsilon * groundtruth_batch + (1 - epsilon) * chanvese_batch
            )  # [batchsize, channels, height, width]
            intermediate_batch.requires_grad = True
            assert intermediate_batch.isnan().sum() == 0

            # apply the neural network
            groundtruth_out = NN(groundtruth_batch)  # [batchsize]
            chanvese_out = NN(chanvese_batch)  # [batchsize]
            intermediate_out = NN(intermediate_batch)  # [batchsize]

            # calculate the loss
            wasserstein_loss = (groundtruth_out - chanvese_out).mean()  # [1]
            assert wasserstein_loss.isnan().sum() == 0

            # Set 'create_graph=True' so we can backprop a function of the
            # gradient (--> second derivatives).
            # REVIEW: Maybe we should use "retain_graph"?
            gradient = torch.autograd.grad(
                intermediate_out.sum(), intermediate_batch, create_graph=True
            )[0]
            # --> [batchsize, channels, height, width]
            assert gradient.isnan().sum() == 0

            gradient_loss = (
                F.relu(gradient.square().sum((1, 2, 3)).sqrt() - 1).square().mean()
            )

            loss = wasserstein_loss + mu * gradient_loss  # [1]
            assert loss.isnan().sum() == 0
            print(
                "Wasserstein loss: {:.2f}\t Gradient Loss: {:.2f}\t Min Groundtruth: {:.2f}\t Max Chan-Vese: {:.2f}".format(
                    wasserstein_loss.item(),
                    gradient_loss.item(),
                    groundtruth_out.min().item(),
                    chanvese_out.max().item(),
                )
            )

            # backprop step
            # no need to zero the gradients of the intermediate point, since it
            # is reinitialised each batch
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            assert NN.conv1.weight.isnan().sum() == 0

            print(f"Epoch {j} done.")

    if save_path is not None:
        torch.save(NN.state_dict(), save_path)

    return NN.to("cpu")
