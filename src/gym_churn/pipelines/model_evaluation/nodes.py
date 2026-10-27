"""Model evaluation nodes."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def evaluate_model(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    metrics = {"auc": float(auc)}
    return y_pred, y_proba, metrics


def plot_confusion_matrix(y_test, y_pred):
    fig = plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    classes = ["Active", "Churned"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(np.round(cm[i, j], 2)),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_test, y_proba, metrics):
    auc = metrics["auc"]
    fig = plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_cols):
    fig = plt.figure(figsize=(13, 5))
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    importances.plot(kind="bar", title="Feature Importance")
    plt.ylabel("importance")
    plt.tight_layout()
    return fig
