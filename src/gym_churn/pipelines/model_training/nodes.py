"""
Section 2 of notebooks/01_clean_notebook - model training.

The functions are copied as-is from the notebook.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def split_data(members, feature_cols, test_size, random_state):
    X = members[feature_cols]
    y = members["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators, max_depth, random_state):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def build_test_set(X_test, y_test):
    test_set = X_test.copy()
    test_set["churned"] = y_test.values
    return test_set
