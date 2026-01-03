import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, ax=None):
    """
    Plots a heatmap confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title = "Confusion Matrix (Normalized)"
    else:
        fmt = "d"
        title = "Confusion Matrix (Counts)"

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)

    return ax


def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Generates predictions, prints classification report, and returns metrics.

    Args:
        model: Trained Keras model.
        X_test, y_test: Test data.
        class_names: List of string labels for classes.

    Returns:
        results: Dictionary containing y_pred and metrics.
    """
    print("[INFO] Generating predictions...")

    # 1. Predict
    # Softmax output: shape (n_samples, n_classes)
    y_prob = model.predict(X_test)

    # Convert probabilities to class indices (0, 1, 2, 3)
    y_pred = np.argmax(y_prob, axis=1)

    # Ensure y_test is 1D
    if isinstance(y_test, pd.DataFrame):
        y_true = y_test.values.flatten()
    else:
        y_true = y_test

    # 2. Metrics
    acc = accuracy_score(y_true, y_pred)
    print(f"\n--- Overall Accuracy: {acc:.4f} ---\n")

    print("--- Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": acc,
        "report": report,
        "report": report,
    }
