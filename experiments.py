import numpy as np
import matplotlib.pyplot as plt
from helper_data_preprocessing import (
    build_poly,
    CustomStandardScaler,
    random_undersampling,
)
from helper_training import holdout, EarlyStopper


############  Logistic Regression  #################### DONE
def sigmoid(x):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return np.piecewise(
        x,
        [x > 0],
        [lambda i: 1 / (1 + np.exp(-i)), lambda i: np.exp(i) / (1 + np.exp(i))],
    )


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """

    N = y.shape[0]
    y_proba = sigmoid(np.dot(tx, w))

    # Clippig for numerical stability
    clip_y_proba = np.clip(y_proba, 1e-10, 1 - 1e-10)
    clip_1_y_proba = np.clip(1 - y_proba, 1e-10, 1 - 1e-10)

    log_likelihood = y * np.log(clip_y_proba) + (1 - y) * np.log(clip_1_y_proba)
    loss = -1 * np.sum(log_likelihood) / N
    return loss


def calculate_gradient(y, tx, w):
    """compute the gradient of loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a vector of shape (D, 1)
    """
    N = y.shape[0]
    y_pred = np.dot(tx, w)
    gradient = np.dot(tx.T, sigmoid(y_pred) - y) / N
    return gradient


def reg_logistic_regression(
    y, tx, lambda1, lambda2, initial_w, max_iters, gamma, batch_size, plot=False
):
    """
    Perform regularized logistic regression.

    Args:
        y: shape=(N, 1)
        tx: shape=(N, D)
        lambda1: L1 regularization parameter (float).
        lambda2: L2 regularization parameter (float).
        initial_w: shape=(D, 1)
        max_iters: maximum iterations (int).
        gamma: Learning rate for gradient descent (float).
        batch_size: Size of mini-batches for stochastic gradient descent (int).
        plot: Whether to plot and display loss curves, default is False.

    Returns:
        (w, loss): A tuple containing the optimized weights, shape=(D, 1), and the final loss (float).
    """

    # initialise starting weights
    w = initial_w

    # get holdout set for early stopping condition
    train_holdout_ratio = 0.90
    tx, tx_ho, y, y_ho = holdout(tx, y, train_holdout_ratio)

    early_stopper = EarlyStopper(
        patience=10, min_delta=0.001
    )  # stop if the holdout error > (best validation error + min_delta) for the last PATIENCE iterations

    # get indices for minibatch draws later
    indices = np.arange(len(y))

    # prepare arrays to store losses for plotting
    ho_losses = []
    tr_losses = []

    for n_iter in range(max_iters):
        # get batch
        batch_indices = np.random.choice(indices, size=batch_size)
        minibatch_y, minibatch_tx = y[batch_indices], tx[batch_indices]

        # get regularized gradients
        gradient = (
            calculate_gradient(minibatch_y, minibatch_tx, w)
            + ((2 * lambda1 * w) / batch_size)
            + (lambda2 * np.sign(w)) / batch_size
        )

        # Update the weights
        w -= gamma * gradient

        # calculate and store holdout loss
        ho_loss = calculate_loss(y_ho, tx_ho, w)
        ho_losses.append(ho_loss)

        # calculate and store training loss
        if plot:
            tr_loss = calculate_loss(minibatch_y, minibatch_tx, w)
            tr_losses.append(tr_loss)

        # check for early stopping
        if early_stopper.early_stop(ho_loss):
            # print("Early stoppping reached after {} iterations".format(n_iter))
            break

    # Compute the loss for the final value of the weights
    loss = np.array(calculate_loss(y, tx, w))

    # plot losses and provide summary stats to debug
    if plot:
        plt.plot(range(len(ho_losses)), ho_losses, c="blue", label="Holdout Loss")
        plt.plot(range(len(tr_losses)), tr_losses, c="red", label="Minibatch Loss")
        plt.legend()
        plt.show()
        print(
            "Min of Losses: {}, Max of Losses: {}, Number of Nas in Losses: {}".format(
                min(ho_losses), max(ho_losses), np.isnan(ho_losses).sum()
            )
        )

    return w, loss


def logistic_regression_prob_prediction(x, w):
    """
    Predict the probability of a sample belonging to a class using the logistic function.

    Args:
        x: shape=(N, D)
        w: shape=(D, 1)

    Returns:
        prob: shape=(N, 1) predicted probabilities.
    """
    return sigmoid(np.dot(x, w))


def reg_logistic_regression_experiment(
    y_tr, y_te, x_tr, x_te, lambda1, lambda2, max_iters, gamma, batch_size, plot
):
    """
    Perform a regularized logistic regression experiment.

    Args:
        y_tr: shape=(N1, 1)
        y_te: shape=(N2, 1)
        x_tr: shape=(N1, D)
        x_te: shape=(N2, D)
        lambda1: L1 regularization parameter (float).
        lambda2: L2 regularization parameter (float).
        max_iters: maximum iterations (int).
        gamma: Learning rate for gradient descent (float).
        batch_size: Size of mini-batches for stochastic gradient descent (int).
        plot: Whether to plot and display loss curves, default is False.

    Returns:
        (opt_w, loss_tr, loss_te, pred_tr, pred_te): the optimized weights, loss on the training set, loss on the testing set,
        predicted probabilities for the training set, and predicted probabilities for the testing set (all scalars).
    """

    # scale training and testing data
    scaler = CustomStandardScaler()
    # scaler = CustomNormalizer()
    x_tr = scaler.fit_transform(x_tr)
    x_te = scaler.transform(x_te)

    # add vector of 1s for intercept
    x_tr = np.c_[np.ones((x_tr.shape[0], 1)), x_tr]
    x_te = np.c_[np.ones((x_te.shape[0], 1)), x_te]

    initial_w = np.zeros((x_tr.shape[1], 1)).astype(np.longdouble)

    # Training: find the optimal parameters
    opt_w, loss_tr = reg_logistic_regression(
        y_tr, x_tr, lambda1, lambda2, initial_w, max_iters, gamma, batch_size, plot
    )
    pred_te = logistic_regression_prob_prediction(x_te, opt_w)

    # Testing: checking performance of test set
    loss_te = calculate_loss(y_te, x_te, opt_w)
    pred_tr = logistic_regression_prob_prediction(x_tr, opt_w)

    return opt_w, loss_tr, loss_te, pred_tr, pred_te


def ensemble_logistic_regression_experiment(
    y_tr, y_te, x_tr, x_te, ensemble_params, row_subsample=0, col_subsample=1
):
    """
    Perform an ensemble experiment of regularized logistic regression.

    Args:
        y_tr: shape=(N1, 1)
        y_te: shape=(N2, 1)
        x_tr: shape=(N1, D)
        x_te: shape=(N2, D)
        ensemble_params: A list of model parameters for each ensemble member.
        row_subsample: The ratio of rows to subsample, default is 0 (float).
        col_subsample: The ratio of columns to subsample, default is 1 (float).

    Returns:
        tuple: A tuple containing the ensemble-averaged predicted probabilities for the testing set, ensemble-optimized weights,
        ensemble-averaged training loss, ensemble-averaged testing loss, zero (not described in the code),
        median outcome based on majority voting, and ensemble-averaged test predictions.
    """

    ensemble_opt_w = []
    losses_tr, losses_te = np.array([]), np.array([])
    preds_te = []

    # Iterating through each model parameter
    for model_param in ensemble_params:
        print(model_param)

        if row_subsample != 0:
            N, C = x_tr.shape
            bootstrap_sample = np.random.choice(N, int(row_subsample * N), replace=True)
            column_sample = np.random.choice(
                C, int(col_subsample * C), replace=False
            )  # column sampling this way does not work, so its turned off
            x_tr_ = x_tr[np.ix_(bootstrap_sample, column_sample)]
            x_te_ = x_te[:, column_sample]
            y_tr_ = y_tr[np.ix_(bootstrap_sample)]
        else:
            x_tr_ = x_tr
            y_tr_ = y_tr
            x_te_ = x_te

        # extract data modification parameters
        majority_to_minority_ratio, minority_multiplier = model_param[-1]
        degree = model_param[-2]
        log_transform = model_param[-3]

        # only access model_param
        parameters = model_param[:-3]

        # upsample if needed
        if majority_to_minority_ratio != 0:
            x_tr_, y_tr_ = random_undersampling(
                x_tr_, y_tr_, majority_to_minority_ratio, minority_multiplier
            )

        log_x_tr = np.log(x_tr_ + 1)
        log_x_te = np.log(x_te_ + 1)

        # build polynomial feature matrix
        x_tr_ = build_poly(x_tr_, degree)
        x_te_ = build_poly(x_te_, degree)

        # TODO: add log transform
        if log_transform:
            x_tr_ = np.column_stack((x_tr_, log_x_tr))
            x_te_ = np.column_stack((x_te_, log_x_te))

        opt_w, loss_tr, loss_te, pred_tr, pred_te = reg_logistic_regression_experiment(
            y_tr_, y_te, x_tr_, x_te_, *parameters
        )

        # Saving metrics for each ensemble model
        ensemble_opt_w.append(opt_w)
        losses_tr = np.append(losses_tr, loss_tr)
        losses_te = np.append(losses_te, loss_te)

        preds_te.append(pred_te)

    # Computing average results
    avg_loss_tr = np.mean(losses_tr)
    avg_loss_te = np.mean(losses_te)

    # calculate test predictions - train predictions not possible, as we have different shapes due to undersampling

    # calculate average of predicted probability
    preds_te = np.column_stack(preds_te)
    avg_pred_te = np.mean(preds_te, axis=1)

    # do majority voting as well
    individual_classifications = (preds_te >= 0.5).astype(int)
    median_outcome = (np.mean(individual_classifications, axis=1) > 0.5).astype(
        int
    )  # predict yes if over half the classifiers said yes

    return (
        avg_pred_te,
        ensemble_opt_w,
        avg_loss_tr,
        avg_loss_te,
        0,
        avg_pred_te,
        median_outcome,
    )
