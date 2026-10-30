"""
Model evaluation pipeline: trained model + test split -> metrics + plots.
"""

from kedro.pipeline import Node, Pipeline  # noqa: F401

from .nodes import (  # noqa: F401
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)


def create_pipeline(**kwargs) -> Pipeline:
    # TODO: one Node per function in nodes.py.
    #
    # Hint 1: evaluate_model returns THREE values -> list of three output names.
    # Hint 2: plot_roc_curve takes the whole metrics dict (see its signature).
    # Hint 3: inputs here are the outputs of model_training (churn_model, X_test, y_test).
    return Pipeline([])
