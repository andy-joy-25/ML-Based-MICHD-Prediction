import numpy as np
from helper_data_preprocessing import build_poly, random_undersampling
from helper_metrics import compute_metrics


class EarlyStopper:
    """
    Implements Early Stopping

    Attributes:
        patience: (int) Maximum number of iterations with no decrease in validation loss after which the training will be stopped
        min_delta: (float) Threshold for the increase in the validation loss to qualify as a non-improvement, i.e. an upward change of greater than min_delta will qualify as no improvement.
        counter: (int) Number of iterations during which the validation loss has not decreased
        min_validation_loss: (float) Minimum value of the validation loss

    """

    def __init__(self, patience=1, min_delta=0):
        """
        Constructor for the EarlyStopper class

        Arguments:
            patience: (int) Maximum number of iterations with no decrease in validation loss after which the training will be stopped
            min_delta: (float) Minimum increase in the validation loss to qualify as a non-improvement, i.e. an upward change of greater than min_delta will qualify as no improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        """
        Instance method to check for early stopping, i.e., to determine whether or not the training needs to be stopped midway

        Arguments:
            validation_loss: (float) The validation loss for the current iteration

        Returns:
            (bool) True if the training needs to be stopped and False otherwise
        """
        # Check whether the current validation loss is lower than the (until then) minimum validation loss
        if validation_loss < self.min_validation_loss:
            # Update the value of the minimum validation loss with the value of the current validation loss
            self.min_validation_loss = validation_loss
            # Don't increment the counter
            self.counter = 0

        # Check whether the increase in the validation loss has crossed the the threshold of 'min_delta'
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            # Increment the counter
            self.counter += 1

            # Check whether we have reached our level of patience
            if self.counter >= self.patience:
                # Stop the training
                return True

        # Don't stop the training
        return False


def holdout(x, y, train_holdout_ratio, random_gen_seed=1):
    """
    Generate the holdout validation set

    Args:
        x: (numpy array of shape=(N,D)) [N is the number of samples, D is the number of features]
        y: (numpy array of shape=(N,1))
        train_holdout_ratio: (float) The ratio of the number of samples in the training set to that in the holdout validation set
        random_gen_seed: (int) Seed for random number generation

    Returns:
        x_tr, x_ho, y_tr, y_ho: (numpy arrays) The training (x_tr and y_tr) and holdout validation (x_tr and y_tr) sets

    """
    # Set the seed for random number generation
    np.random.seed(random_gen_seed)

    # Randomize the ordering of samples
    random_row_order = np.random.permutation(x.shape[0])
    x, y = x[random_row_order], y[random_row_order]

    # Generate the training and holdout sets as per the given ratio
    num_tr_samples = int(np.floor(train_holdout_ratio * x.shape[0]))
    x_tr, x_ho, y_tr, y_ho = (
        x[:num_tr_samples],
        x[num_tr_samples:],
        y[:num_tr_samples],
        y[num_tr_samples:],
    )

    return x_tr, x_ho, y_tr, y_ho


def build_k_indices(y, k_fold, seed):
    """
    Build k indices for k-fold.

    Args:
        y:      (numpy array of shape=(N,1)) [N is the number of samples]
        k_fold: (int) K in K-fold, i.e. the number of folds
        seed: (int) Seed for random number generation

    Returns:
        A 2D numpy array of shape = (k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    # Determine the interval for splitting the data into k folds
    interval = int(num_row / k_fold)
    # Randomize the order of the samples in the Cross Validation (CV) set
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    # Determine the indices of the samples in each CV set
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(
    model,
    x,
    y,
    k_indices,
    k,
    model_parameters,
    degree,
    log_transform,
    majority_to_minority_ratio,
    minority_multiplier,
    classification_threshold,
):
    """
    Returns the optimal weights, training and testing metrics for a fold corresponding to k_indices

    Args:
        model: (function) the ML model for classification
        x:          (numpy array of shape=(N,D)) [N is the number of samples, D is the number of features]
        y:          (numpy array of shape=(N,1))
        k_indices:  2D numpy array returned by build_k_indices()
        k: (int) the kth fold of Cross Validation
        model_parameters: (list) parameters for the ML model
        degree: (int) degree for polynomial expansion
        log_transform: (bool) if True, we log transform the features, if False, we don't
        majority_to_minority_ratio: (int) Y in the Ratio Y:1 for undersampling
        minority_multiplier: (int) the oversampling parameter for the minority class
        classification_threshold: (float) the threshold for classification


    Returns:
        opt_w: (numpy array of shape=(D,1)) optimal weights
        acc_te: (float) Test Accuracy
        acc_tr: (float) Train Accuracy
        loss_te: (float) Test Loss
        loss_tr: (float) Train Loss
        f1_te: (float) Test F1-Score
        f1_tr: (float) Train F1-Score
    """

    # get the kth subgroup
    te_idx_net = k_indices[k]
    tr_idx_net = np.delete(k_indices, k, axis=0).flatten()
    x_tr, x_te, y_tr, y_te = (
        x[tr_idx_net, :],
        x[te_idx_net, :],
        y[tr_idx_net],
        y[te_idx_net],
    )

    # resample if specified
    if majority_to_minority_ratio != 0:
        x_tr, y_tr = random_undersampling(
            x_tr, y_tr, majority_to_minority_ratio, minority_multiplier
        )

    log_x_tr = np.log(x_tr + 1)
    log_x_te = np.log(x_te + 1)

    # build polynomial feature matrix
    x_tr = build_poly(x_tr, degree)
    x_te = build_poly(x_te, degree)

    # add a log transform if specified
    if log_transform:
        x_tr = np.column_stack((log_x_tr, x_tr))
        x_te = np.column_stack((log_x_te, x_te))

    # train the model
    opt_w, loss_tr, loss_te, pred_tr, pred_te = model(
        y_tr, y_te, x_tr, x_te, *model_parameters
    )

    # calculate training and testing metrics on this current fold
    acc_te, acc_tr, f1_te, f1_tr = compute_metrics(
        y_te, pred_te, y_tr, pred_tr, classification_threshold
    )

    return opt_w, acc_te, acc_tr, loss_te, loss_tr, f1_te, f1_tr


def best_parameters_selection(model, x, y, k_fold, model_parameters_net, seed=1):
    """
     Cross Validation over all the parameters (Helpful in selecting the best model parameters)

     Args:
         model:      (function) the ML model for classification
         x:          (numpy array of shape=(N,D)) [N is the number of samples, D is the number of features]
         y:          (numpy array of shape=(N,1))
         k_fold:      (int) the number of folds
         model_parameters_net:  list of list of all possible combinations of model parameters [[],[],[],...]
         seed: (int) Seed for random number generation

    Returns:
         results : list of the averages of the metrics
         results : list of the standard deviations of the metrics
    """

    # split data into k folds
    k_indices = build_k_indices(y, k_fold, seed)

    results = []
    results_std = []

    for (
        model_parameters
    ) in (
        model_parameters_net
    ):  # Grid search over the different sets of model parameters (including polynomial degree)
        metrics = []

        # extract data modification parameters
        classification_threshold = model_parameters[-1]
        majority_to_minority_ratio, minority_multiplier = model_parameters[-2]
        degree = model_parameters[-3]
        log_transform = model_parameters[-4]

        # only access model_parameters
        parameters = model_parameters[:-4]

        # print("Running model for degree {}, majority to minority ratio of {}, minority_multiplier of {}, classification threshold of {} and the following model parameters {}".format(degree, majority_to_minority_ratio, minority_multiplier, classification_threshold, parameters))

        for k in range(k_fold):
            #  function returns all relevant metrics along with optimal weights
            w_opt, acc_te, acc_tr, loss_te, loss_tr, f1_te, f1_tr = cross_validation(
                model,
                x,
                y,
                k_indices,
                k,
                parameters,
                degree,
                log_transform,
                majority_to_minority_ratio,
                minority_multiplier,
                classification_threshold,
            )
            # print("(Threshold: {}) Test F1: {} \n Train F1: {}".format(classification_threshold, f1_te, f1_tr))
            # keep track of model performance on split
            metrics_row = [acc_te, acc_tr, loss_te, loss_tr, f1_te, f1_tr]
            metrics.append(metrics_row)  # this is a list of lists

        # calculate mean and standard deviation across folds

        metrics = np.array(metrics, dtype=np.float64)  # convert to numpy array

        metrics_avg = np.mean(metrics, axis=0)  # average over columns
        metrics_std = np.std(metrics, axis=0)  # average over columns

        metrics = np.hstack((metrics_avg, metrics_std))

        # append results
        results.append(metrics_avg)
        results_std.append(metrics_std)

    return results, results_std
