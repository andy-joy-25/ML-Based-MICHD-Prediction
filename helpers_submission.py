import numpy as np
import os
from helper_data_preprocessing import prepare_data


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids: (list,np.array) indices
        y_pred: (list,np.array) predictions on data corresponding to the indices
        name: (str) name of the CSV file to be created
    """
    import csv

    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    # Create the CSV file with the IDs and the corresponding predictions
    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def create_file(
    file,
):
    """
    Create a CSV file

    Args:
        file: (str) name of the CSV file to be created
    """
    import csv

    # Headers for the CSV file
    headers = [
        "Model",
        "Params",
        "Data",
        "Test F1",
        "Train F1",
        "Test Accuracy",
        "Train Accuracy",
        "Test Loss",
        "Train Loss",
    ]
    # Create the CSV file
    with open(file, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=headers)
        writer.writeheader()
    return


def append_row(file, name, params, row_metrics, data="No Information"):
    """
    Append a row to the CSV file

    Args:
        file (str): name of the CSV file to be created
        name: (str) name of the ML model used for classification
        params: list of model parameters
        row_metrics: numpy array of the model's metrics
        data: (str) the dataset used for training the model
    """
    # Print the metrics
    print(row_metrics)

    import csv

    # Create the CSV file
    with open(file, "a", newline="") as csvfile:
        # Headers for the CSV file
        headers = [
            "Model",
            "Params",
            "Data",
            "Test F1",
            "Train F1",
            "Test Accuracy",
            "Train Accuracy",
            "Test Loss",
            "Train Loss",
        ]

        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=headers)

        # Extract the different metrics so as to store them in separate columns
        c1, c2, c3, c4, c5, c6 = row_metrics.tolist()
        writer.writerow(
            {
                "Model": str(name),
                "Params": str(params),
                "Data": str(data),
                "Test F1": float(c5),
                "Train F1": float(c6),
                "Test Accuracy": float(c1),
                "Train Accuracy": float(c2),
                "Test Loss": float(c3),
                "Train Loss": float(c4),
            }
        )


def load_data(DATA_PATH, path, labels_path="train_labels.npy"):
    """
    Loads the data

    Args:
        DATA_PATH: (str) The main path to the data
        path: (str) The path to the training data with the ids
        labels_path: (str) The path to the training data's labels

    Returns:
        the prepared data for training and testing (ref. prepare_data())
    """

    labels = np.load(os.path.join(DATA_PATH, labels_path))
    data_ids = np.load(os.path.join(DATA_PATH, path))
    # Extract the features and store the ids separately since we don't require them for training
    data = data_ids[:, 1:]
    ids = data_ids[:, 0]
    return prepare_data(data, labels, test_ratio=0.2, subsample=False)


def ensemble_submission(
    y_train: np.array,
    X_train: np.array,
    X_test: np.array,
    params: list,
    threshold: float = 0.5,
):
    """
    Generate the final predictions for the test set using the ensemble learning approach

    y_train: (numpy array of shape=(N1,1))
    X_train: (numpy array of shape=(N1,D)) [N1 is the number of training samples, D is the number of features]
    X_test: (numpy array of shape=(N2,D)) [N2 is the number of test samples, D is the number of features]
    params: list of list of parameters for the ensemble (each list in the main list contains the parameters for an ensemble member)
    threshold: (float) the classification threshold

    Returns:
        pred_te: (numpy array of shape=(N2,1)) The test set predictions
        test_ids: (numpy array of shape=(N2,1)) The test set IDs
    """

    from experiments import ensemble_logistic_regression_experiment

    X_train = X_train[:, 1:]  # feature vectors
    test_ids = X_test[:, 0]  # ID vector
    X_test = X_test[:, 1:]

    # reshape and cast to floats
    y_train = y_train.reshape(-1, 1)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    assert all(
        i in [0, 1] for i in y_train
    ), "Model training requires labels in the form of [0; 1]"

    # create random y_test to feed to the function - since the argument is required and here there is no y_test
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
        y_train, random_y_test, X_train, X_test, params
    )

    # Map the ensemble-averaged predicted probabilities to the predictions (classes for the test set)
    predictions = np.where(proba > threshold, 1, 0)

    # remap predictions to -1 and 1
    pred_te = np.where(predictions == 0, -1, predictions)

    return pred_te, test_ids
