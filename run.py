import numpy as np
import os
from A1_Data_Processing import Data
from helpers_submission import ensemble_submission, create_csv_submission


# define seed for reproducibility
np.random.seed(42)  # global seed

# define constants and experiments settings
DATA_PATH = r"C:\data_repository\ML\raw_data"  # path to your data
SUBMISSION_NAME = "submission_6.csv"
PARAMS = [0.05, 0.5, 1000, 0.05, 1028, False, True, 2, (3, 2)]
NUM_MODELS = 10
BOOTRAPPING = 0.8  # 0.8 is the best submssion so far
THRESHOLD = 0.5


def main():
    # path to data files
    train_path = os.path.join(DATA_PATH, "x_train.csv")
    test_path = os.path.join(DATA_PATH, "x_test.csv")
    train_label_path = os.path.join(DATA_PATH, "y_train.csv")

    # data processing
    data = Data(train_path, test_path, train_label_path)
    train_labels = data.train_labels
    train_processed, test_processed, _ = data.process_data()

    # define paramters for each model - here just the same parameters for each model in NUM_MODELS
    ensemble_params = [PARAMS] * NUM_MODELS

    # train and predict
    predictions, ids = ensemble_submission(
        train_labels, train_processed, test_processed, ensemble_params, THRESHOLD
    )

    # create csv submission
    create_csv_submission(ids, predictions, SUBMISSION_NAME)
    return


if __name__ == "__main__":
    main()
