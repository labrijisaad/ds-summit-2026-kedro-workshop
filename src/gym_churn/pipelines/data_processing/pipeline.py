"""
Data processing pipeline: raw members + check-ins -> model input table.

This pipeline is 1/4 wired as a worked example. Use the same pattern -
one Node() per function - to add the other three nodes below.
"""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    build_checkin_features,  # noqa: F401
    build_target,
    clean_members,  # noqa: F401
    encode_categoricals,  # noqa: F401
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # ---- WORKED EXAMPLE -----------------------------------------------
            # Note the three things that make this a Kedro node:
            #   - inputs are *dataset names* (from the catalog) or "params:xxx"
            #   - outputs is the dataset name this function produces
            #   - name is only used by `kedro run --nodes <name>` and the graph
            Node(
                func=build_target,
                inputs=["members", "checkins", "params:churn_window_days"],
                outputs="members_with_target",
                name="build_target_node",
            ),
            # ---- TODO ---------------------------------------------------------
            # Add the three remaining nodes. The output of one is the input of
            # the next - that is how Kedro figures out the execution order.
            #
            #   build_checkin_features(members_with_target, checkins, churn_window_days)
            #       -> members_with_features
            #
            #   clean_members(members_with_features, fill_values)
            #       -> members_clean
            #
            #   encode_categoricals(members_clean, cat_cols)
            #       -> model_input_table
            #
            # Reminder: parameters go in as "params:fill_values", "params:cat_cols".
        ]
    )
