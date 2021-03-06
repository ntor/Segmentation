import numpy as np
import torch
import torch.nn.functional as F

"""
the regularisation parameter lambda CANNOT be initialised in the same way for segmentation as for denoising
"""
"""
the function unreg_mini in Sebastian's paper shouldn't need to be redone here, as chanvese should already be a good starting point for reconstruction
"""


"""
must recalculate data fitting term in pytorch, so we can compute gradients
"""
"""
I THINK THIS IS THE IMPLEMENTATION MIKE WANTED
"""


def data_fitting(
    chanvese_batch,
    noisy_batch,
    lambda_chanvese=1,
    threshold=0.5,
    c1=None,
    c2=None,
    alpha=None,
):
    """
    chanvese_batch & noisy_batch must a torch.tensor (ideally on gpu) of size [batchsize, 1, height, width]
    noisy_batch contains the corresponding noisy images to chanvese_batch
    threshold is used to find the segmentation boundary (for the purpose of calculating c1, c2) from chanvese_batch
    """
    assert chanvese_batch.size() == noisy_batch.size()
    assert chanvese_batch.size(1) == 1  # require greyscale image, i.e. only one channel

    # calculate c1, c2 implicity from u
    """
    WE DO NOT want to backprop along c1, c2 when performing the reconstruction (algorithm 2), only relevant when c1, c2 are calculated implicitly
    Actually not a problem, since no operation below is going to preserve requires_grad = true
    """
    if c1 == None:
        c1 = (noisy_batch * (chanvese_batch > threshold)).mean((1, 2, 3))  # [batchsize]
    if c2 == None:
        c2 = (noisy_batch * (chanvese_batch <= threshold)).mean(
            (1, 2, 3)
        )  # [batchsize]

    chanvese_term = lambda_chanvese * (
        (noisy_batch - c1.unsqueeze(1).unsqueeze(2).unsqueeze(3)).square()
        - (noisy_batch - c2.unsqueeze(1).unsqueeze(2).unsqueeze(3)).square()
    )  # [batchsize, 1, height, width]

    # calculate alpha implicity from u, lambda, c1, and c2?
    """
    WE DO NOT want to backprop along alpha when performing the reconstruction (algorithm 2), only relevant when alpha is calculated implicitly
    hence .detach() below
    """
    if alpha == None:
        alpha = (
            chanvese_term.detach().abs().max(3)[0].max(2)[0].max(1)[0]
        )  # [batchsize]

    penality_term = 2 * (
        (chanvese_batch - 0.5).abs() - 1
    )  # [batchsize, 1, height, width]
    penality_term = penality_term * (penality_term > 0)  # [batchsize, 1, height, width]

    """
    integral over domain is just done by taking the mean, should just correspond to scaling lambda_reg accordingly in reconstruct (below)
    """
    datafitting_term = (
        chanvese_term * chanvese_batch
        + alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3) * penality_term
    ).mean(
        (1, 2, 3)
    )  # [batchsize]

    return datafitting_term  # [batchsize]


"""
ALGORITHM 2:
simultaneously perform a number of gradient descent steps on a full batch of chanvese segmentations, or already partialy reconstructed segmentations
(both would take the argument chanvese_batch below)
noisy_batch contains the corresponding noisy images to chanvese_batch (for the purpose of the datafitting term above)
"""


def reconstruct(
    chanvese_batch, noisy_batch, NN, lambda_reg, epsilon, reconstruction_steps=1
):
    """
    chanvese_batch & noisy_batch must be a torch.tensor of size [batchsize, channels, height, width]
    NN is the learnt regulariser
    lambda_reg is how much we weight the regularising term (not the datafitting term) when reconstructing the solution according to algorithm 2
    """
    device = next(NN.parameters()).device  # trick to find device NN is stored on
    reconstructed_batch = chanvese_batch.to(
        device
    ).detach()  # transfer chanvese_batch to same device NN is stored on, detach just incase
    noisy_batch_copy = noisy_batch.to(
        device
    )  # transfer noisy_batch to same device NN is stored on

    for i in range(reconstruction_steps):
        reconstructed_batch.requires_grad = True  # set requires_grad to True, gradients are initialised at zero, and entire backprop graph will be recreated (not the most efficient way, as autograd graph has to be recreated each time)

        """
        data_fitting function not yet implemented
        """
        datafitting = data_fitting(reconstructed_batch, noisy_batch_copy)  # [batchsize]
        regularising = NN(reconstructed_batch)  # [batchsize]

        error = datafitting + lambda_reg * regularising  # [batchsize]
        error = error.sum()  # [1], trick to compute all gradients in one go

        gradients = torch.autograd.grad(error, reconstructed_batch)[0]
        reconstructed_batch = (
            reconstructed_batch - epsilon * gradients
        ).detach()  # detaching from previous autograd which also sets requires_grad to False

    return (
        reconstructed_batch.to(chanvese_batch.device),
        noisy_batch,
    )  # send back to original device


"""
a quick function for evaluating the quality of the reconstructed segmentation according to the L2 difference between it and groundtruth
"""


def quality(reconstructed_batch, groundtruth_batch):
    """
    reconstructed_batch, and groundtruth_batch must be torch.tensors of the same size [batchsize, channels, height, width]
    """
    return (
        (reconstructed_batch - groundtruth_batch).square().sum((1, 2, 3)).sqrt()
    )  # [batchsize]


"""
analogue of Sebastian's function log_minimum (except without storing any data in logs), which keeps reconstructing solutions until their quality (as defined above, i.e. requiring knowledge of the ground truth) no longer keeps decreasing

idea is to use this to evaluate performance of NN
"""


def minimum(chanvese_batch, noisy_batch, groundtruth_batch, NN, lambda_reg, epsilon):

    assert chanvese_batch.size() == noisy_batch.size()
    assert chanvese_batch.size() == groundtruth_batch.size()
    assert chanvese_batch.device == noisy_batch.device
    assert chanvese_batch.device == groundtruth_batch.device
    batchsize = chanvese_batch.size(0)
    device = chanvese_batch.device

    todo_mask = torch.ones([batchsize], dtype=torch.bool, device=device)
    chanvese_todo = chanvese_batch
    noisy_todo = noisy_batch
    groundtruth_todo = groundtruth_batch
    quality_prev = torch.full([batchsize], float("inf"), device=device)
    minimum_batch = torch.empty_like(chanvese_todo)
    final_quality = torch.empty_like(quality_prev)
    steps = torch.zeros_like(quality_prev)

    while todo_mask.sum():
        steps += todo_mask

        chanvese_todo = reconstruct(chanvese_todo, noisy_todo, NN, lambda_reg, epsilon)[0]
        quality_new = quality(chanvese_todo, groundtruth_todo)
        done_mask = quality_new > quality_prev

        done_mask_unravel = torch.zeros_like(todo_mask).masked_scatter(
            todo_mask, done_mask
        )

        minimum_batch.masked_scatter_(
            done_mask_unravel.unsqueeze(1)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(minimum_batch.size()),
            chanvese_todo.masked_select(
                done_mask.unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .expand(chanvese_todo.size())
            ),
        )

        final_quality.masked_scatter_(
            done_mask_unravel, quality_new.masked_select(done_mask)
        )
        todo_mask.masked_fill_(done_mask_unravel, False)

        chanvese_todo = chanvese_todo.masked_select(
            done_mask.logical_not()
            .unsqueeze(1)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(chanvese_todo.size())
        ).view(
            -1, chanvese_batch.size(1), chanvese_batch.size(2), chanvese_batch.size(3)
        )
        noisy_todo = noisy_todo.masked_select(
            done_mask.logical_not()
            .unsqueeze(1)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(noisy_todo.size())
        ).view(
            -1, chanvese_batch.size(1), chanvese_batch.size(2), chanvese_batch.size(3)
        )
        groundtruth_todo = groundtruth_todo.masked_select(
            done_mask.logical_not()
            .unsqueeze(1)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand(groundtruth_todo.size())
        ).view(
            -1, chanvese_batch.size(1), chanvese_batch.size(2), chanvese_batch.size(3)
        )
        quality_prev = quality_new.masked_select(done_mask.logical_not())

    return (
        minimum_batch,
        final_quality,
        steps,
    )  # outputs the final (optimal) reconstruction, their corresponding quality, and the reconstruction steps required
