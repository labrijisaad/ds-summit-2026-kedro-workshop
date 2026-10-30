"""Model training nodes.

The function signatures are given - you fill in the bodies by copying from
section 2 of notebooks/01_clean_notebook. Nothing clever to invent here.
"""

from sklearn.ensemble import RandomForestClassifier  # noqa: F401
from sklearn.model_selection import train_test_split  # noqa: F401


def split_data(members, feature_cols, test_size, random_state):
    """80/20 split -> X_train, X_test, y_train, y_test (in that order)."""
    # TODO: copy the body from the clean notebook.
    raise NotImplementedError


def train_model(X_train, y_train, n_estimators, max_depth, random_state):
    """Fit a RandomForestClassifier and return it."""
    # TODO: copy the body from the clean notebook.
    raise NotImplementedError


def build_test_set(X_test, y_test):
    """Glue X_test and y_test back together so it can be saved as one CSV."""
    # TODO: copy the body from the clean notebook.
    raise NotImplementedError
