"""Model evaluation nodes.

Function signatures given - fill in the bodies from section 3 of
notebooks/01_clean_notebook.
"""

import matplotlib.pyplot as plt  # noqa: F401
import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve  # noqa: F401


def evaluate_model(model, X_test, y_test):
    """Return (y_pred, y_proba, metrics) where metrics is {"auc": float}."""
    # TODO: copy the body from the clean notebook.
    raise NotImplementedError


def plot_confusion_matrix(y_test, y_pred):
    """Return a matplotlib Figure. Save handled by the catalog."""
    # TODO: copy the body from the clean notebook.
    raise NotImplementedError


def plot_roc_curve(y_test, y_proba, metrics):
    """Return a matplotlib Figure.

    Note: this takes the whole `metrics` dict, not a bare float - a node can
    only consume datasets produced by other nodes, not an attribute of one.
    """
    # TODO: copy the body from the clean notebook (read `auc` from metrics["auc"]).
    raise NotImplementedError


def plot_feature_importance(model, feature_cols):
    """Return a matplotlib Figure."""
    # TODO: copy the body from the clean notebook (return only the figure).
    raise NotImplementedError
