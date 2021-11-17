"""Loss function modules."""

import numpy as np

from lib.activations import Softmax
from lib.network_base import Module


class CrossEntropyLoss(Module):
    """Computes the softmax and then cross-entropy loss."""

    def __init__(self):
        super().__init__()
        self.softmax = Softmax()

    def forward(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """Compute the cross entropy, mean over batch size.

        Args:
            preds: Model predictions with shape (batch_size, num_classes)
            labels: Ground truth labels with shape (batch_size, num_classes)

        Returns:
            Cross-entropy loss.
        """
        assert len(preds.shape) == 2, (
            "Predictions should be of shape (batchsize, num_classes) "
            f"but are of shape {preds.shape}")
        assert len(labels.shape) == 2, (
            "Labels should be of shape (batchsize, num_classes) "
            f"but are of shape {labels.shape}")
        assert preds.shape == labels.shape, (
            "Predictions and labels should be of same shape but are "
            f"of shapes {preds.shape} and {labels.shape}")
        preds = self.softmax(preds)
        self.input_cache = preds, labels

        # compute the mean loss over the batch
        return -np.sum(np.log(preds[labels == 1])) / len(preds)

    def backward(self, _: None = None) -> np.ndarray:
        """Calculate the backward pass of the cross-entropy loss. Remember that the softmax is included in the
        forward pass, so you will have to include it in the gradient as well.

        Args:
            _: Unused gradient, we introduce the argument to have a unified interface with
                other Module objects. This simplifies code for gradient checking.
                We don't need this arg since there will not be a layer after the loss layer.

        Returns:
            The gradient of this module with shape (batch_size, num_classes).
        """
        a, y = self.input_cache

        # START TODO ################
        # calculate gradient
        grad = (1/len(a)) * (a-y)
        # END TODO ##################

        return grad
