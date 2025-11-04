import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def f1_score(pred, y_true):
    """
    Calculate the F1-Score for binary classification.

    Args:
        pred: shape=(N, 1)
        y_true: shape=(N, 1)

    Returns:
        float: the F1 score
    """

    pred = pred.flatten()
    y_true = y_true.flatten()

    true_pos = np.sum((pred == 1) & (y_true == 1))
    false_pos = np.sum((pred == 1) & (y_true == 0))
    false_neg = np.sum((pred == 0) & (y_true == 1))

    prec = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    rec = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)


def classify_prediction(pred, threshold):
    """
    Classify predicted probabilities based on a threshold.

    Args:
        pred: shape=(N, 1)
        threshold: float

    Returns:
        classified predictions: shape=(N, 1)
    """
    return (pred >= threshold).astype(int)


def compute_metrics(y_te, pred_te, y_tr, pred_tr, classification_threshold):
    """
    Compute various classification metrics for binary classification.

    Args:
        y_te: shape=(N1, 1)
        pred_te: shape=(N1, 1)
        y_tr: shape=(N2, 1)
        pred_tr: shape=(N2, 1)
        classification_threshold: float

    Returns:
        (acc_te, acc_tr, f1_te, f1_tr): Testing accuracy, training accuracy, testing f1-score and training f1-score (all floats)
    """

    pred_tr = classify_prediction(pred_tr, classification_threshold)
    pred_te = classify_prediction(pred_te, classification_threshold)

    if y_te.ndim > 1:
        y_te = y_te[:, 0]
    if pred_te.ndim > 1:
        pred_te = pred_te[:, 0]
    if y_tr.ndim > 1:
        y_tr = y_tr[:, 0]
    if pred_tr.ndim > 1:
        pred_tr = pred_tr[:, 0]

    assert y_te.ndim == 1 and y_tr.ndim == 1 and pred_te.ndim == 1 and pred_tr.ndim == 1

    assert y_te.shape[0] == pred_te.shape[0]
    assert y_tr.shape[0] == pred_tr.shape[0]

    acc_te = np.mean(y_te == pred_te)
    acc_tr = np.mean(y_tr == pred_tr)

    f1_te = f1_score(pred_te, y_te)
    f1_tr = f1_score(pred_tr, y_tr)

    return acc_te, acc_tr, f1_te, f1_tr


def compute_confusion_matrix(y_true, pred, classification_threshold=0.5):
    """
    Compute an visualize confusion matrix for binary classification.

    Args:
        y_true: shape=(N, 1)
        pred: shape=(N, 1)
        classification_threshold: float
    """

    if y_true.ndim > 1:
        y_true = y_true[:, 0]
    if y_true.ndim > 1:
        pred = pred[:, 0]

    assert y_true.ndim == 1 and pred.ndim == 1

    pred = classify_prediction(pred, classification_threshold)

    true_pos = np.sum((pred == 1) & (y_true == 1))
    true_neg = np.sum((pred == 0) & (y_true == 0))
    false_pos = np.sum((pred == 1) & (y_true == 0))
    false_neg = np.sum((pred == 0) & (y_true == 1))

    matrix = np.array([[true_pos, false_pos], [false_neg, true_neg]])
    print(matrix)

    plt.figure(figsize=(8, 8))
    ax = sns.heatmap(
        matrix,
        vmax=np.max(matrix),
        vmin=np.min(matrix),
        annot=True,
        cmap="crest",
        cbar=False,
        fmt="d",
    )
    ax.set(xlabel="Actuals", ylabel="Predicted")
    ax.set_xticklabels([1, 0])
    ax.set_yticklabels([1, 0])
    return


def threshold_plot(proba, y_true):
    """
    Plot the F1 score against different classification thresholds.

    Args:
        proba: shape=(N, 1)
        y_true: shape=(N, 1)
    """

    classification_thresholds = np.linspace(0, 1, 100)
    random_y = np.zeros(y_true.shape)
    f_score = []
    for classification_threshold in classification_thresholds:
        metrics = compute_metrics(
            y_true, proba, random_y, random_y, classification_threshold
        )
        f_score.append(metrics[2])  # metrics[2] -> test f-score

    # Create a line plot to visualize how F1 score changes with different thresholds
    plt.figure(figsize=(10, 6))
    plt.plot(classification_thresholds, f_score, marker="o")
    plt.xlabel("Classification Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs. Classification Threshold")
    plt.grid(True)
    plt.show()
