#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:31:48 2023

@author: MagicaL
"""

import numpy as np


def compute_mse_loss(y, tx, w):
    """
    Compute the Mean Squared Error (Loss function for Linear Regression)

    Args:
        y: numpy array of shape=(N, ) [N is the number of samples]
        tx: numpy array of shape=(N,D) [D is the number of features]
        w: numpy array of shape=(D,). [The vector of model parameters]

    Returns:
        mse = scalar denoting the value of the loss corresponding to the input parameters w.
    """

    # Compute the error
    e = y - np.matmul(tx, w)

    # Compute the MSE
    mse = np.matmul(e.T, e) / (2 * tx.shape[0])

    return mse


def compute_mse_gradient(y, tx, w):
    """
    Compute the Gradient of the MSE at w.

    Args:
        y: numpy array of shape=(N, ) [N is the number of samples]
        tx: numpy array of shape=(N,D) [D is the number of features]
        w: numpy array of shape=(D, ). [The vector of model parameters]

    Returns:
        grad = numpy array of shape (D, ) (same shape as w), containing the gradient of the MSE at w.
    """

    # Compute the error
    e = y - np.matmul(tx, w)

    # Compute the gradient of the MSE
    grad = -(1 / y.shape[0]) * np.dot(tx.T, e)

    return grad


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    The Gradient Descent (GD) algorithm for Linear Regression using MSE.

    Args:
        y: numpy array of shape=(N, ) [N is the number of samples]
        tx: numpy array of shape=(N,D) [D is the number of features]
        initial_w: numpy array of shape=(D, ) [The initial guess (or the initialization) for the model parameters]
        max_iters: scalar denoting the total number of iterations of GD
        gamma: scalar denoting the learning rate

    Returns:
        w: numpy array of shape (D, ) containing the model parameters obtained after the last iteration of GD
        loss: scalar denoting the loss value obtained after the last iteration of GD
    """

    # Set w to the intiial value of the weights
    w = initial_w

    # Run GD for max_iters number of iterations
    for n_iter in range(max_iters):
        # Compute the gradient
        grad = compute_mse_gradient(y, tx, w)

        # Update the weights
        w = w - gamma * grad

    # Compute the loss (MSE) for the final value of the weights
    loss = np.array(compute_mse_loss(y, tx, w))

    return w, loss


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.

    Args:
        y: numpy array of shape=(N, ) [N is the number of samples]
        tx: numpy array of shape=(N,D) [D is the number of features]
        batch_size: scalar denoting the number of examples (datapoints) in each mini-batch
        num_batches: scalar denoting the number of mini-batches to be generated
        shuffle: boolean specifying whether to randomly shuffle the data (shuffle = True) or not (shuffle = False)

    Details:
        Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
        Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
        Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Returns:
        A generator object (an iterator) which gives 'num_batches' number of mini-batches (numpy arrays) of size `batch_size` (as much as possible)

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """

    # Store the number of training labels as the size of the data
    data_size = len(y)

    if shuffle:
        # Shuffle to randomize the order of the training datapoints
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx

    # Generate the mini-batches
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    The Stochastic Gradient Descent (SGD) algorithm for Linear regression using MSE.

    Args:
        y: numpy array of shape=(N, ) [N is the number of samples]
        tx: numpy array of shape=(N,D) [D is the number of features]
        initial_w: numpy array of shape=(D, ) [The initial guess (or the initialization) for the model parameters]
        max_iters: scalar denoting the total number of iterations of SGD
        gamma: scalar denoting the learning rate

    Returns:
        w: numpy array of shape (D, ) containing the model parameters obtained after the last iteration of SGD
        loss: scalar denoting the loss value obtained after the last iteration of SGD
    """

    # Set w to the intiial value of the weights
    w = initial_w

    # Set the batch size to 1 for SGD
    batch_size = 1

    # Run SGD for max_iters number of iterations
    for n_iter in range(max_iters):
        # Generate the mini-batches
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # Compute the stochastic gradient (compute the usual gradient but with mini-batches now)
            grad = compute_mse_gradient(minibatch_y, minibatch_tx, w)

        # Update the weights
        w = w - gamma * grad

    # Compute the loss (MSE) for the final value of the weights
    loss = np.array(compute_mse_loss(y, tx, w))

    return w, loss


def least_squares(y, tx):
    """
    Calculate the least squares solution.

    Args:
        y: numpy array of shape (N,) [N is the number of samples]
        tx: numpy array of shape (N,D) [D is the number of features]

    Returns:
        w: numpy array of shape (D, ) containing the optimal weights
        loss: scalar denoting the loss value obtained using the optimal weights

    """

    # Solve the linear system for least squares to get the optimal weights
    w = np.linalg.solve(np.matmul(tx.T, tx), np.matmul(tx.T, y))

    # Compute the loss (MSE) for the optimal value of the weights
    loss = np.array(compute_mse_loss(y, tx, w))

    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Implement ridge regression.

    Args:
        y: numpy array of shape (N,) [N is the number of samples]
        tx: numpy array of shape (N,D) [D is the number of features]
        lambda_: scalar denoting the L2 regularization parameter

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar denoting the loss value obtained using the optimal weights

    """

    # Solve the linear system for least squares to get the optimal weights
    w = np.linalg.solve(
        np.matmul(tx.T, tx) + 2 * y.shape[0] * lambda_ * np.identity(tx.shape[1]),
        np.matmul(tx.T, y),
    )

    # Compute the loss (MSE) for the optimal value of the weights
    loss = np.array(compute_mse_loss(y, tx, w))

    return w, loss


def sigmoid(t):
    """
    Apply the sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array containing the result of applying sigmoid on t
    """

    return 1 / (1 + np.exp(-t))


def compute_log_loss(y, tx, w):
    """
    Compute the loss using negative log likelihood.

    Args:
        y:  numpy array of shape=(N, ) [N is the number of samples]
        tx: numpy array of shape=(N, D) [D is the number of features]
        w:  numpy array of shape=(D, ) [The vector of model parameters]

    Returns:
        loss: scalar denoting the non-negative loss
    """
    # Compute the total number of samples (datapoints)
    N = y.shape[0]

    # Compute the numpy array of predictions
    y_pred = np.dot(tx, w)

    # Compute the log likelihood
    log_likelihood = y * np.log(sigmoid(y_pred)) + (1 - y) * np.log(1 - sigmoid(y_pred))

    # Compute the loss
    loss = -1 * np.sum(log_likelihood) / N

    return loss


def compute_log_loss_gradient(y, tx, w):
    """
    Compute the gradient of the logistic loss.

    Args:
        y:  numpy array of shape=(N, ) [N is the number of samples]
        tx: numpy array of shape=(N, D) [D is the number of features]
        w:  numpy array of shape=(D, ) [The vector of model parameters]

    Returns:
        grad = numpy array of shape (D, ) (same shape as w), containing the gradient of the log loss at w.
    """

    # Compute the total number of samples (datapoints)
    N = y.shape[0]

    # Compute the numpy array of predictions
    y_pred = np.dot(tx, w)

    # Compute the gradient of the log loss
    gradient = np.dot(tx.T, sigmoid(y_pred) - y) / N

    return gradient


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Perform Logistic Regression using Gradient Descent (GD)

    Args:
        y: numpy array of shape (N, ) [N is the number of samples]
        tx: numpy array of shape (N, D) [D is the number of features]
        initial_w: numpy array of shape (D, ) [The initial guess (or the initialization) for the model parameters]
        max_iters: scalar denoting the total number of iterations of GD
        gamma: scalar denoting the learning rate

    Returns:
        w: numpy array of shape (D, ) containing the model parameters obtained after the last iteration of GD
        loss: scalar denoting the loss value obtained after the last iteration of GD
    """

    # Set w to the intiial value of the weights
    w = initial_w

    # Run GD for 'max_iters' number of iterations
    for n_iter in range(max_iters):
        # Compute the gradient
        gradient = compute_log_loss_gradient(y, tx, w)

        # Update the weights
        w -= gamma * gradient

    # Compute the loss for the final value of the weights
    loss = np.array(compute_log_loss(y, tx, w))

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Perform Regularized Logistic Regression using Gradient Descent (GD)

    Args:
        y: numpy array of shape (N, ) [N is the number of samples]
        tx: numpy array of shape (N, D) [D is the number of features]
        lambda_: scalar denoting the L2 regularization parameter
        initial_w: numpy array of shape (D, ) [The initial guess (or the initialization) for the model parameters]
        max_iters: scalar denoting the total number of iterations of GD
        gamma: scalar denoting the learning rate

    Returns:
        w: numpy array of shape (D, ) containing the model parameters obtained after the last iteration of GD
        loss: float - scalar denoting the loss value obtained after the last iteration of GD
    """

    # Set w to the intiial value of the weights
    w = initial_w

    # Run GD for 'max_iters' number of iterations
    for n_iter in range(max_iters):
        # Compute the gradient of the loss and the regularization term
        gradient = compute_log_loss_gradient(y, tx, w) + (2 * lambda_ * w)

        # Update the weights
        w -= gamma * gradient

    # Compute the loss for the final value of the weights
    loss = np.array(compute_log_loss(y, tx, w))

    return w, loss
