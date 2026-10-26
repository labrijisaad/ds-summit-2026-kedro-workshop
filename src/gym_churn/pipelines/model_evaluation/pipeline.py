"""
Model evaluation pipeline: trained model + test split -> metrics and plots.
"""

from kedro.pipeline import Node, Pipeline  # noqa: F401

from .nodes import (  # noqa: F401
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # TODO: one Node per function in nodes.py.
            # Hint: the inputs here are the outputs of the model_training pipeline.
            # Hint: check each function's signature - plot_roc_curve takes the whole
            #       metrics dict instead of a bare float.
        ]
    )
