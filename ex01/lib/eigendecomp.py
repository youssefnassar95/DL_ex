"""Eigendecomposition functions."""

import numpy as np


def get_matrix_from_eigdec(e: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Restore the original square symmetric matrix from eigenvalues and eigenvectors after eigenvalue decomposition.

    Args:
        e: The vector of eigenvalues with shape (N).
        V: The matrix with eigenvectors as columns with shape (N, N).

    Returns:
        The original matrix used for eigenvalue decomposition with shape (N, N)
    """
    # START TODO #################
    v_transpose = np.transpose(V)  # transpose the eigenvector Q
    e_diag = np.diag(e)  # form a matrix N x N with the diagonal is the eigenvalues
    A = np.matmul(np.matmul(V, e_diag), v_transpose)
    return A
    raise NotImplementedError
    # END TODO ###################


def get_euclidean_norm(v: np.ndarray) -> np.ndarray:
    """Compute the euclidean norm of a vector.

    Args:
        v (np.ndarray): The input vector with shape (vector_length).

    Returns:
        The euclidean norm of the vector.
    """
    # START TODO #################
    # do NOT use np.linalg.norm
    euclid_norm = np.sqrt((v ** 2).sum())
    return euclid_norm
    raise NotImplementedError
    # END TODO ###################


def get_dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute dot product of two vectors.

    Args:
        v1: First input vector with shape (vector_length)
        v2: Second input vector with shape (vector_length)

    Returns:
        Dot product result.
    """
    assert len(v1.shape) == len(v2.shape) == 1 and v1.shape == v2.shape, \
        f"Input vectors must be 1-dimensional and have the same shape, but have shapes {v1.shape} and {v2.shape}"
    # START TODO #################
    dot_product = (v1 * v2).sum()
    return dot_product
    raise NotImplementedError
    # END TODO ###################


def get_inverse(e: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Compute the inverse of a square symmetric matrix A given its eigenvectors and eigenvalues.

    Args:
        e: The vector of eigenvalues with shape (N).
        V: The matrix with eigenvectors as columns with shape (N, N).

    Returns:
        The inverse of A (i.e. the matrix with given eigenvalues/vectors) with shape (N, N).
    """
    # START TODO #################
    # Do not use np.linalg.inv, otherwise you will get no points.
    v_transpose = np.transpose(V)
    e_inv = e ** (-1)
    e_inv_diag = np.diag(e_inv)
    A_inv = np.matmul(np.matmul(V, e_inv_diag), v_transpose)
    return A_inv
    raise NotImplementedError
    # END TODO ###################
