"""
Model training pipeline: model input table -> train/test split -> trained model.
"""

from kedro.pipeline import Node, Pipeline  # noqa: F401

from .nodes import build_test_set, split_data, train_model  # noqa: F401


def create_pipeline(**kwargs) -> Pipeline:
    # TODO: one Node per function in nodes.py.
    #
    # Hint 1: split_data returns FOUR values -> outputs must be a list of four names.
    # Hint 2: train_model needs X_train and y_train - those are the outputs of split_data.
    # Hint 3: churn_model is the dataset model_evaluation will consume.
    return Pipeline([])
