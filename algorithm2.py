import numpy as np
import torch
import torch.nn.functional as F

"""
the regularisation parameter lmb CANNOT be initialised in the same way for segmentation as for denoising
"""
"""
the function unreg_mini in Sebastian's paper shouldn't need to be redone here, as chanvese should already be a good starting point for reconstruction
"""


"""
must recalculate data fitting term in pytorch, so we can compute gradients
"""
"""
THIS IS JUST A QUICK PLACEHOLDER FOR NOW
(honestly really confused by the difference between z and u in this context, so not sure how to implement the data fitting term)
"""
def data_fitting(chanvese_batch, lmb1 = 1, lmb2 = 1, alpha = None, c1 = None, c2 = None):
    """
    chanvese_batch must a torch.tensor (ideally on gpu) of size [batchsize, 1, height, width]
    """
    assert chanvese_batch.size(1) == 1 #require greyscale image, i.e. only one channel
    
    #calculate c1, c2 implicity from u?
    if c1 == None:
        c1 = ############
    if c2 == None:
        c2 = ############
    
    #not sure, but I don't think we want to backprop along c1, c2 when performing the reconstruction (algorithm 2), only relevant when c1, c2 are calculated implicitly
    c1 = c1.detach()
    c2 = c2.detach()
    
    chanvese_term = ############
    
    #calculate alpha implicity from u, lambda1, lambda2, c1, and c2?
    if alpha == None:
        alpha = ############
    
    #not sure, but I don't think we want to backprop along alpha when performing the reconstruction (algorithm 2), only relevant when alpha is calculated implicitly
    alpha = alpha.detach()
    
    penality_term = ############
    
    return chanvese_term * chanvese_batch + alpha * penality_term #[batchsize]


"""
ALGORITHM 2:
simultaneously perform a number of gradient descent steps on a full batch of chanvese segmentations, or already partialy reconstructed segmentations
(both would take the argument chanvese_batch below)
"""
def reconstruct(chanvese_batch, NN, lmb, epsilon, reconstruction_steps = 1):
    """
    chanvese_batch must be a torch.tensor of size [batchsize, channels, height, width]
    NN is the learnt regulariser
    """
    reconstructed_batch = chanvese_batch.to(next(NN.parameters()).device) #transfer chanvese_batch to same device NN is stored on
    
    for i in range(reconstruction_steps):
        reconstructed_batch.requires_grad = True #set requires_grad to True, gradients are initialised at zero, and entire backprop graph will be recreated (not the most efficient way, as autograd graph has to be recreated each time)
        
        """
        data_fitting function not yet implemented
        """
        datafitting = data_fitting(reconstructed_batch) #[batchsize]
        regularising = NN(reconstructed_batch) #[batchsize]
        
        error = datafitting + lmb * regularising #[batchsize]
        error = error.sum() #[1], trick to compute all gradients in one go
        
        gradients = torch.autograd.grad(error, reconstructed_batch)[0]
        reconstructed_batch = (reconstructed_batch - epsilon * gradients).detach() #detaching from previous autograd which also sets requires_grad to False
    
    return reconstructed_batch.to(chanvese_batch.device) #send back to original device


"""
a quick function for evaluating the quality of the reconstructed segmentation according to the L2 difference between it and groundtruth
"""
def quality(reconstructed_batch, groundtruth_batch):
    """
    reconstructed_batch, and groundtruth_batch must be torch.tensors of the same size [batchsize, channels, height, width]
    """
    return (reconstructed_batch - groundtruth_batch).square().sum((1, 2, 3)).sqrt() #[batchsize]


"""
analogue of Sebastian's function log_minimum (except without storing any data in logs), which keeps reconstructing solutions until their quality (as defined above, i.e. requiring knowledge of the ground truth) no longer keeps decreasing
"""
def minimum(chanvese_batch, groundtruth_batch, NN, lmb, epsilon):
    
    assert chanvese_batch.size() == groundtruth_batch.size()
    assert chanvese_batch.device == groundtruth_batch.device
    batchsize = chanvese_batch.size(0)
    device = chanvese_batch.device
    
    todo_mask = torch.ones([batchsize], dtype = torch.bool, device = device)
    chanvese_todo = chanvese_batch.transpose(0, -1)
    groundtruth_todo = groundtruth_batch.transpose(0, -1)
    quality_prev = torch.full([batchsize], float('inf'), device = device)
    minimum_batch = torch.empty_like(chanvese_todo)
    final_quality = torch.empty_like(quality_prev)
    steps = torch.zeros_like(quality_prev)
    
    while todo_mask.sum():
        
        chanvese_todo = reconstruct(chanvese_todo, NN, lmb, epsilon)
        quality_new = quality(chanvese_todo, groundtruth_todo)
        done_mask = quality_new > quality_prev
        
        done_mask_unravel = torch.zeros_like(todo_mask).masked_scatter(todo_mask, done_mask)
        
        minimum_batch.masked_scatter_(done_mask_unravel, chanvese_todo.masked_select(done_mask))
        final_quality.masked_scatter_(done_mask_unravel, quality_new.masked_select(done_mask))
        todo_mask.masked_fill_(done_mask_unravel, False)
        steps.index_put_(done_mask_unravel.nonzero(as_tuple = True), torch.tensor([1], dtype = torch.float, device = device), accumulate = True)
        
        chanvese_todo = chanvese_todo.masked_select(done_mask.logical_not).view(chanvese_todo.size(0), chanvese_todo.size(1), chanvese_todo.size(2), -1)
        groundtruth_todo = groundtruth_todo.masked_select(done_mask.logical_not).view(groundtruth_todo.size(0), groundtruth_todo.size(1), groundtruth_todo.size(2), -1)
        quality_prev =  quality_new.masked_select(done_mask.logical_not)
    
    return minimum_batch.transpose(0, -1), final_quality, steps #outputs the final (optimal) reconstruction, their corresponding quality, and the reconstruction steps required