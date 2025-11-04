import numpy as np
import os
from helper_data_preprocessing import build_poly, random_undersampling
from helper_metrics import compute_confusion_matrix, compute_metrics


def test_evaluation(
    X_train, y_train, X_test, y_test, model, model_parameters, conf_mat=True
):
    """
    Evaluate the model

    Args:
        X_train: (numpy array of shape=(N1,D)) [N1 is the number of training samples, D is the number of features]
        y_train: (numpy array of shape=(N1,1))
        X_test: (numpy array of shape=(N2,D)) [N2 is the number of test samples, D is the number of features]
        y_test: (numpy array of shape=(N2,1))
        model: (function) the ML model for classification
        model_parameters: (list) parameters for the ML model
        conf_mat: (bool) if True, compute the confusion matrix, otherwise, don't
    """

    # extract data modification parameters
    classification_threshold = model_parameters[-1]
    majority_to_minority_ratio, minority_multiplier = model_parameters[-2]
    degree = model_parameters[-3]
    log_transform = model_parameters[-4]

    # only access model_parameters
    parameters = model_parameters[:-4]

    # resample if needed
    if majority_to_minority_ratio != 0:
        X_train, y_train = random_undersampling(
            X_train, y_train, majority_to_minority_ratio, minority_multiplier
        )

    log_x_tr = np.log(X_train + 1)
    log_x_te = np.log(X_test + 1)

    # build polynomial feature matrix
    X_train = build_poly(X_train, degree)
    X_test = build_poly(X_test, degree)

    # add log transforms if requested
    if log_transform:
        X_train = np.column_stack((X_train, log_x_tr))
        X_test = np.column_stack((X_test, log_x_te))

    # train the model
    opt_w, loss_tr, loss_te, pred_tr, pred_te = model(
        y_train, y_test, X_train, X_test, *parameters
    )

    pred_tr = pred_tr[:, 0]
    pred_te = pred_te[:, 0]

    # Compute the confusion matrix
    if conf_mat:
        compute_confusion_matrix(
            y_test,
            pred_te,
        )

    # Compute the matrics
    metrics = compute_metrics(
        y_test, pred_te, y_train, pred_tr, classification_threshold
    )

    METRICS = ["Test Accuracy", "train Accuracy", "Test F1", "Train F1"]

    # Print the metrics
    for name, val in zip(METRICS, metrics):
        print("{:>40}: {}".format(name, round(val, 5)))

    return


def predict_unseen(
    data_path, train_path, test_path, label_path, model, model_parameters
):
    """
    Generate predictions for the test set

    data_path: (str) The main path to the data
    train_path: (str) The path to the train data
    test_path: (str) The path to the test data
    label_path: (str) The path to the train labels
    model: (function) the ML model for classification
    model_parameters: (list) parameters for the ML model

    Returns:
        pred_te: (numpy array of shape=(N2,1)) The test set predictions
        test_ids: (numpy array of shape=(N2,1)) The test set IDs
    """

    # load everything
    X_train = np.load(os.path.join(data_path, train_path))
    X_train = X_train[:, 1:]
    X_test = np.load(os.path.join(data_path, test_path))
    test_ids = X_test[:, 0]
    X_test = X_test[:, 1:]

    y_train = np.load(os.path.join(data_path, label_path))

    # reshape and cast to floats
    y_train = y_train.reshape(-1, 1)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # extract data modification parameters
    majority_to_minority_ratio, minority_multiplier = model_parameters[-1]
    degree = model_parameters[-2]
    log_transform = model_parameters[-3]

    # only access model_parameters
    parameters = model_parameters[:-3]

    # resample if needed
    if majority_to_minority_ratio != 0:
        X_train, y_train = random_undersampling(
            X_train, y_train, majority_to_minority_ratio, minority_multiplier
        )

    log_x_tr = np.log(X_train)
    log_x_te = np.log(X_test)

    # build polynomial feature matrix
    X_train = build_poly(X_train, degree)
    X_test = build_poly(X_test, degree)

    # adding log transforms
    if log_transform:
        X_train = np.column_stack((np.log(X_train + 1), log_x_tr))
        X_test = np.column_stack((np.log(X_test + 1), log_x_te))

    # train the model
    random_y_test = np.zeros((X_test.shape[0], 1))
    opt_w, loss_tr, loss_te, pred_tr, pred_te = model(
        y_train, random_y_test, X_train, X_test, *parameters
    )

    pred_tr = pred_tr[:, 0]

    # confusion matrix on training set
    compute_confusion_matrix(y_train, pred_tr)

    # remap predictions to -1 and 1
    pred_te = np.where(pred_te == 0, -1, pred_te)

    return pred_te, test_ids


def final_predictions_ensemble(
    data_path, train_path, test_path, label_path, PARAMS, threshold=0.5
):
    """
    Generate the final predictions for the test set using the ensemble learning approach

    data_path: (str) The main path to the data
    train_path: (str) The path to the train data
    test_path: (str) The path to the test data
    label_path: (str) The path to the train labels
    PARAMS: list of list of parameters for the ensemble (each list in the main list contains the parameters for an ensemble member)
    threshold: (float) the classification threshold

    Returns:
        pred_te: (numpy array of shape=(N2,1)) The test set predictions
        test_ids: (numpy array of shape=(N2,1)) The test set IDs
    """

    from experiments import ensemble_logistic_regression_experiment

    # load everything
    X_train = np.load(os.path.join(data_path, train_path))
    X_train = X_train[:, 1:]
    X_test = np.load(os.path.join(data_path, test_path))
    test_ids = X_test[:, 0]
    X_test = X_test[:, 1:]
    y_train = np.load(os.path.join(data_path, label_path))

    # reshape and cast to floats
    y_train = y_train.reshape(-1, 1)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    random_y_test = np.zeros((X_test.shape[0], 1))

    # Run the ensemble model and obtain the ensemble-averaged predicted probabilities for the test set
    (
        proba,
        _,
        _,
        _,
        ensemble_pred_tr,
        ensemble_pred_te,
        median_votes,
    ) = ensemble_logistic_regression_experiment(
        y_train, random_y_test, X_train, X_test, PARAMS
    )

    # Map the ensemble-averaged predicted probabilities to the predictions (classes for the test set)
    predictions = np.where(proba > threshold, 1, 0)

    # remap predictions to -1 and 1
    pred_te = np.where(predictions == 0, -1, predictions)

    return pred_te, test_ids
