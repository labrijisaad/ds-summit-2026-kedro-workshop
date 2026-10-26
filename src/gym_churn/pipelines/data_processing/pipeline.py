"""
Data processing pipeline: raw members + check-ins -> model input table.
"""

from kedro.pipeline import Node, Pipeline  # noqa: F401

from .nodes import (  # noqa: F401
    build_checkin_features,
    build_target,
    clean_members,
    encode_categoricals,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # TODO: one Node per function in nodes.py, in any order -
            # Kedro works out the execution order from inputs and outputs.
            #
            # Node(
            #     func=build_target,
            #     inputs=["members", "checkins", "params:churn_window_days"],
            #     outputs="members_with_target",
            #     name="build_target_node",
            # ),
        ]
    )
