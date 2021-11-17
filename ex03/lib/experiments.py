"""Experiments with the models."""

from lib.network_base import Module
from lib.losses import CrossEntropyLoss


def backward_pass(model: Module, loss_fn: CrossEntropyLoss, lr: float) -> None:
    """After the forward pass was performed and the loss calculated,
    this function performs the backward pass of the model.

    Args:
        model: Module to train.
        loss_fn: Loss function to use for training, in this case CrossEntropyLoss.
        lr: Learning rate to use for training.

    """
    # START TODO ################
    # Perform a backward pass on the model to obtain the gradients and
    gradient = loss_fn.backward()
    # then use them to update the parameters of the network
    for i in model.parameters():
        i.data -= i.grad
    # END TODO ##################
