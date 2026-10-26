"""
Model training pipeline: model input table -> train/test split -> trained model.
"""

from kedro.pipeline import Node, Pipeline  # noqa: F401

from .nodes import build_test_set, split_data, train_model  # noqa: F401


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # TODO: one Node per function in nodes.py.
            # Hint: a function that returns several values needs a list of outputs.
        ]
    )
