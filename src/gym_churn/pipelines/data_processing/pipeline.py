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
            Node(
                func=build_target,
                inputs=["members", "checkins", "params:churn_window_days"],
                outputs="members_with_target",
                name="build_target_node",
            ),
            # TODO: add the remaining 3 nodes following the same pattern.
            # Node(
            #     func=build_checkin_features,
            #     inputs=["members_with_target", "checkins", "params:churn_window_days"],
            #     outputs="members_with_features",
            #     name="build_checkin_features_node",
            # ),
        ]
    )
