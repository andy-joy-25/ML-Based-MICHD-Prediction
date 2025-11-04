#### Script to train the final classifier to create the submission

from helper_inference import predict_unseen, final_predictions_ensemble
from helpers_submission import create_csv_submission
from experiments import reg_logistic_regression_experiment


PATH = r"C:\data_repository\ML\raw_data"  # path to your data
SUBMISSION_NAME = "submission_5.csv"
ENSEMBLE = True
BEST_PARAMS = [1.0, 1.0, 1000, 0.05, 1028, True, False, 4, (3, 2)]
BEST_PARAMS2 = [0.1, 0.0, 1000, 0.05, 1028, True, False, 3, (3, 2)]
BEST_PARAMS3 = [
    [0.1, 0.0, 1000, 0.05, 1028, False, False, 3, (3, 2)],
    [1.0, 1.0, 1000, 0.05, 1028, False, False, 3, (3, 3)],
    [1.0, 0.1, 1000, 0.05, 1028, False, True, 3, (3, 2)],
    [0.1, 0.0, 1000, 0.05, 1028, False, False, 4, (3, 2)],
    [1.0, 1.0, 1000, 0.05, 1028, False, True, 2, (3, 3)],
    [1.0, 0.1, 1000, 0.05, 1028, False, False, 5, (3, 2)],
    [0.1, 0.0, 1000, 0.05, 1028, False, False, 3, (2, 2)],
    [1.0, 1.0, 1000, 0.05, 1028, False, False, 3, (2, 3)],
    [0.1, 0.0, 1000, 0.05, 1028, False, False, 3, (4, 2)],
    [1.0, 1.0, 1000, 0.05, 1028, False, False, 3, (4, 2)],
]

if ENSEMBLE:
    preds, ids = final_predictions_ensemble(
        PATH,
        "train_data_full.npy",
        "test_data_full.npy",
        "train_labels.npy",
        BEST_PARAMS3,
    )
else:
    preds, ids = predict_unseen(
        PATH,
        "train_data_full.npy",
        "test_data_full.npy",
        "train_labels.npy",
        reg_logistic_regression_experiment,
        BEST_PARAMS2,
    )

# create submission
create_csv_submission(ids, preds, SUBMISSION_NAME)
