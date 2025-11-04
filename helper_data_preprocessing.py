import numpy as np
import itertools

# from experiments import calculate_gradient, calculate_loss


def build_poly(x: np.array, degree: int):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)

    >>> build_poly(np.array([0.0, 1.5]), 2)
    array([[1.  , 0.  , 0.  ],
           [1.  , 1.5 , 2.25]])
    """
    out = []
    for col in range(x.shape[1]):
        ad = np.vander(x[:, col], degree + 1, increasing=True)[
            :, 1:
        ]  # drop the column to the power of 0
        out.append(ad)
    return np.column_stack(out)


def lasso_feature_selection(y, tx, initial_w, max_iters, gamma, lambda_):
    """Feature Selection using Lasso Regression (uses GD)

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        lambda_: a scalar, the regularization parameter

    Returns:
        tx: The final numpy array after pruning the features for which the weights were found to be ~ zero
    """

    w = initial_w

    for n_iter in range(max_iters):
        # Compute the gradient
        grad = calculate_gradient(y, tx, w) + (lambda_ * np.sign(w))

        # Update the weights
        w = w - gamma * grad

    # Compute the loss for the final value of the weights
    loss = np.array(calculate_loss(y, tx, w))

    # Find out the indices of the columns for which the weights are ~ zero.
    pruned_feature_idx_net = np.where(abs(w) < 1e-5)[0]

    # Prune those features for which the weights are zero
    if len(pruned_feature_idx_net) != 0:
        tx = np.delete(tx, pruned_feature_idx_net, axis=1)

    return tx


class CustomStandardScaler:
    def __init__(
        self,
    ):
        self.training_means = np.nan
        self.training_sd = np.nan
        return

    def fit(self, data):
        """
        Compute the means and standard deviations of a feature.

        Args:
            data: shape=(N,1)
        """
        self.training_means = np.mean(data, axis=0)
        self.training_sd = np.std(data, axis=0)

    def transform(self, data):
        """
        Standardization with the computed means and standard deviations.

        Args:
            data: shape=(N,1)

        Returns:
            data: Standardized data.
        """
        if np.isnan(self.training_means).any():
            raise ValueError("Scaler must be fit on data before transform is called.")

        return (data - self.training_means) / self.training_sd

    def fit_transform(self, data):
        """
        Fit the scaler on data and immediately transform it.

        Args:
            data: shape=(N,1)

        Returns:
            data: The standardized data.
        """

        self.fit(data)
        return self.transform(data)


def prepare_data(
    X: np.array, y: np.array, test_ratio: float = 0.3, seed: int = 1, subsample=False
) -> (np.array, np.array, np.array, np.array):
    """
    Prepare and split the data into training and testing sets.

    Args:
        X: shape=(N, D)
        y: shape=(N, 1)
        test_ratio: The ratio of splitting the data (float).
        seed: The random seed for data shuffling (int).
        subsample: Subsample ratio (int).

    Returns:
        (X_train, X_test, y_train, y_test): Splitted datasets
    """
    np.random.seed(seed)

    y = y.reshape(-1, 1)  # to fit with the dimensions required

    # split data into training and testing
    indices = np.random.permutation(X.shape[0])
    interval = int(X.shape[0] * test_ratio)
    test_indices = indices[:interval]
    train_indices = indices[interval:]

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    # subsample if required
    if subsample:
        X_train = X_train[:subsample]
        X_test = X_test[:subsample]
        y_train = y_train[:subsample]
        y_test = y_test[:subsample]

    # cast to float32
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # a few sensechecks
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()
    assert np.isfinite(y_train).all()
    assert np.isfinite(y_test).all()
    assert y_test.shape[1] == 1  # dimension for training later

    return X_train, X_test, y_train, y_test


def all_permutations(params):
    """
    Generate all possible permutations of a list of input parameters.

    Args:
        params: A list containing model parameters

    Returns:
        list
    """
    return list(map(list, list(itertools.product(*params))))


def get_majority_minority_indices(y):
    """
    Get the indices of majority and minority classes.

    Args:
        y: numpy array of shape=(N, 1)

    Returns:
        (majority_indices, minority_indices): indices of the majority and minority classes.
    """

    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    if len(class_0_indices) > len(class_1_indices):
        majority_indices = class_0_indices
        minority_indices = class_1_indices
    else:
        majority_indices = class_1_indices
        minority_indices = class_0_indices

    return majority_indices, minority_indices


def random_undersampling(x, y, majority_to_minority_ratio=1.0, minority_multiplier=1):
    """
    Perform random undersampling of the majority class.

    Args:
        x: numpy array of shape=(N,D)
        y: numpy array of shape=(N, 1)
        majority_to_minority_ratio: float
        minority_multiplier: float

    Returns:
        (x_sampled, y_sampled): sampled data
    """

    assert y.shape[0] == x.shape[0]
    majority_indices, minority_indices = get_majority_minority_indices(y)

    minority_multiplier = (
        minority_multiplier - 1
    )  # since if we pass 1, we have 1 times majority class, i.e. no upsampling. if we pass 1.5 we grab 0.5 additional indices etc.

    if minority_multiplier != 0:
        x_minority = x[minority_indices]
        y_minority = y[minority_indices]
        # if int is passed, repeat number of rows without sampling
        if isinstance(minority_multiplier, int):
            additional_x = np.repeat(x_minority, minority_multiplier, axis=0)
            additional_y = np.repeat(y_minority, minority_multiplier, axis=0)

        # if float is passed, randomly sample with replacement
        elif isinstance(minority_multiplier, float):
            additional_rows = minority_multiplier * x_minority.shape[0]
            idx = np.random.choice(x_minority.shape[0], additional_rows, replace=True)
            additional_x = x_minority[idx]
            additional_y = y_minority[idx]

        # append additionally sampled rows
        x = np.vstack([x, additional_x])
        y = np.vstack([y, additional_y])

        # get new majority minority indices
        majority_indices, minority_indices = get_majority_minority_indices(y)

    # Calculate the number of majority class samples
    majority_samples = int(len(minority_indices) * majority_to_minority_ratio)

    random_majority_indices = np.random.choice(
        majority_indices, majority_samples, replace=False
    )
    selected_indices = np.concatenate((random_majority_indices, minority_indices))
    x_sampled = x[selected_indices]
    y_sampled = y[selected_indices]

    # print("Sampled Training Data contains: \n Observations: {} \n Share Minority Class: {}".format(x_sampled.shape[0], np.where(y_sampled[:,0]==1, 1, 0).sum() / y_sampled.shape[0]))

    return x_sampled, y_sampled
