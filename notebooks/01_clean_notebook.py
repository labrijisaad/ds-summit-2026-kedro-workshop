# Databricks notebook source
# MAGIC %md
# MAGIC # Gym Churn - Clean Version
# MAGIC
# MAGIC same data. same model. same results — but now it is readable and testable.
# MAGIC
# MAGIC 📌 *notice the shape of this notebook: three sections, one function per step,*
# MAGIC *and every path and setting declared in a dict instead of buried in the code.*
# MAGIC
# MAGIC *as you read, ask yourself the question you will have to answer in the exercise:*
# MAGIC *if this had to run on a schedule, unattended, with no human clicking cells — what would each piece become?*

# COMMAND ----------

import json
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# COMMAND ----------

# MAGIC %md
# MAGIC ## inputs
# MAGIC
# MAGIC every file this notebook reads, declared in one place.

# COMMAND ----------

INPUTS = {
    "members_path": "../data/01_raw/members.csv",
    "checkins_path": "../data/01_raw/checkins.parquet",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## outputs
# MAGIC
# MAGIC every file this notebook writes, declared in one place.

# COMMAND ----------

OUTPUTS = {
    "test_data_path": "../data/05_model_input/test_data.csv",
    "model_path": "../data/06_models/churn_model.pkl",
    "metrics_path": "../data/08_reporting/metrics.json",
    "confusion_matrix_path": "../data/08_reporting/confusion_matrix.png",
    "roc_curve_path": "../data/08_reporting/roc_curve.png",
    "feature_importance_path": "../data/08_reporting/feature_importance.png",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. data processing
# MAGIC
# MAGIC every choice this section makes is declared in the config below, not hidden in the code.

# COMMAND ----------

PROCESSING_CONFIG = {
    "churn_window_days": 30,
    "fill_values": {
        "contract_type": "Monthly",
    },
    "cat_cols": ["contract_type", "club", "signup_channel"],
}

# COMMAND ----------

# -- NODES ---------------------------------------------------------------------


def build_target(members, checkins, churn_window_days):
    recent = checkins[checkins["days_ago"] <= churn_window_days]
    active_ids = recent["member_id"].unique()
    members = members.copy()
    members["churned"] = (~members["member_id"].isin(active_ids)).astype(int)
    return members


def build_checkin_features(members, checkins, churn_window_days):
    history = checkins[checkins["days_ago"] > churn_window_days]
    checkin_features = (
        history.groupby("member_id")
        .agg(
            past_checkins=("checkin_id", "count"),
            past_avg_duration=("duration_minutes", "mean"),
            past_n_activities=("activity", "nunique"),
        )
        .reset_index()
    )
    members = members.merge(checkin_features, on="member_id", how="left")
    fill_cols = ["past_checkins", "past_avg_duration", "past_n_activities"]
    members[fill_cols] = members[fill_cols].fillna(0)
    return members


def clean_members(members, fill_values):
    members = members.drop_duplicates()
    members["age"] = members["age"].fillna(members["age"].median())
    members["distance_km"] = members["distance_km"].fillna(
        members["distance_km"].median()
    )
    members["contract_type"] = members["contract_type"].fillna(
        fill_values["contract_type"]
    )
    members["club"] = members["club"].str.strip().str.title()
    members["signup_channel"] = members["signup_channel"].str.strip()
    return members


def encode_categoricals(members, cat_cols):
    members = members.copy()
    for col in cat_cols:
        le = LabelEncoder()
        members[col] = le.fit_transform(members[col].astype(str))
    return members


# COMMAND ----------

members = pd.read_csv(INPUTS["members_path"])
checkins = pd.read_parquet(INPUTS["checkins_path"])
print(f"members:  {members.shape}")
print(f"checkins: {checkins.shape}")

# COMMAND ----------

members = build_target(members, checkins, PROCESSING_CONFIG["churn_window_days"])
print(members["churned"].value_counts())
print(f"churn rate: {members['churned'].mean():.1%}")

# COMMAND ----------

members = build_checkin_features(
    members, checkins, PROCESSING_CONFIG["churn_window_days"]
)
members.head()

# COMMAND ----------

members = clean_members(members, PROCESSING_CONFIG["fill_values"])
members.isna().sum()

# COMMAND ----------

members = encode_categoricals(members, PROCESSING_CONFIG["cat_cols"])
members.head()

# COMMAND ----------

print(members.shape)
members.head()

# COMMAND ----------

# MAGIC %md
# MAGIC <details>
# MAGIC <summary>📌 what changed vs the messy notebook — click to reveal</summary>
# MAGIC
# MAGIC - paths live in `INPUTS` — one place to update, not scattered across cells
# MAGIC - `30` is now `churn_window_days` — flows through both `build_target` and `build_checkin_features`
# MAGIC - `"Monthly"` is now named in `PROCESSING_CONFIG`
# MAGIC - each step is a function — testable in isolation
# MAGIC
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. model training
# MAGIC
# MAGIC same idea: the split, the features and the hyperparameters are all declared, never inlined.

# COMMAND ----------

TRAINING_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 200,
    "max_depth": 10,
    "feature_cols": [
        "age",
        "tenure_months",
        "monthly_fee_eur",
        "distance_km",
        "baseline_visits_pm",
        "has_personal_trainer",
        "joined_with_friend",
        "group_classes_pm",
        "contract_type",
        "club",
        "signup_channel",
        "past_checkins",
        "past_avg_duration",
        "past_n_activities",
    ],
}

# COMMAND ----------

# -- NODES ---------------------------------------------------------------------


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


# COMMAND ----------

X_train, X_test, y_train, y_test = split_data(
    members,
    TRAINING_CONFIG["feature_cols"],
    TRAINING_CONFIG["test_size"],
    TRAINING_CONFIG["random_state"],
)
print(f"train: {X_train.shape}")
print(f"test:  {X_test.shape}")
print(f"churn rate train: {y_train.mean():.1%}")
print(f"churn rate test:  {y_test.mean():.1%}")

# COMMAND ----------

test_set = build_test_set(X_test, y_test)

os.makedirs(os.path.dirname(OUTPUTS["test_data_path"]), exist_ok=True)
test_set.to_csv(OUTPUTS["test_data_path"], index=False)
print(f"test set saved: {OUTPUTS['test_data_path']}")

# COMMAND ----------

model = train_model(
    X_train,
    y_train,
    TRAINING_CONFIG["n_estimators"],
    TRAINING_CONFIG["max_depth"],
    TRAINING_CONFIG["random_state"],
)
print("done")

# COMMAND ----------

# MAGIC %md
# MAGIC <details>
# MAGIC <summary>📌 what changed vs the messy notebook — click to reveal</summary>
# MAGIC
# MAGIC - `TRAINING_CONFIG` lives next to the training section — easy to find, easy to change
# MAGIC - `feature_cols` lives in `TRAINING_CONFIG` — single source of truth
# MAGIC - `build_test_set` keeps node logic pure; file writing is explicit and separate
# MAGIC - test set goes to `05_model_input/` — not `01_raw/`
# MAGIC
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. model evaluation

# COMMAND ----------

# -- NODES ---------------------------------------------------------------------


def evaluate_model(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_proba)
    metrics = {"auc": float(auc)}
    return y_pred, y_proba, metrics


def plot_confusion_matrix(y_test, y_pred):
    fig = plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred, normalize="true")

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    classes = ["Active", "Churned"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(np.round(cm[i, j], 2)),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    return fig


def plot_roc_curve(y_test, y_proba, auc):
    fig = plt.figure(figsize=(6, 5))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_cols):
    fig = plt.figure(figsize=(13, 5))
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
        ascending=False
    )
    importances.plot(kind="bar", title="Feature Importance")
    plt.ylabel("importance")
    plt.tight_layout()
    return importances, fig


# COMMAND ----------

y_pred, y_proba, metrics = evaluate_model(model, X_test, y_test)
print(f"AUC: {metrics['auc']:.4f}")

# COMMAND ----------

confusion_matrix_fig = plot_confusion_matrix(y_test, y_pred)

# COMMAND ----------

roc_curve_fig = plot_roc_curve(y_test, y_proba, metrics["auc"])

# COMMAND ----------

importances, feature_importance_fig = plot_feature_importance(
    model, TRAINING_CONFIG["feature_cols"]
)
print(importances)

# COMMAND ----------

os.makedirs(os.path.dirname(OUTPUTS["model_path"]), exist_ok=True)
with open(OUTPUTS["model_path"], "wb") as model_file:
    pickle.dump(model, model_file)
print(f"saved: {OUTPUTS['model_path']}")

os.makedirs(os.path.dirname(OUTPUTS["metrics_path"]), exist_ok=True)
Path(OUTPUTS["metrics_path"]).write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)

confusion_matrix_fig.savefig(OUTPUTS["confusion_matrix_path"], dpi=150)
roc_curve_fig.savefig(OUTPUTS["roc_curve_path"], dpi=150)
feature_importance_fig.savefig(OUTPUTS["feature_importance_path"], dpi=150)

# COMMAND ----------

# MAGIC %md
# MAGIC 📌 metrics, plots, test data, and model are all persisted with explicit paths.

# COMMAND ----------

# MAGIC %md
# MAGIC <details>
# MAGIC <summary>📌 what changed vs the messy notebook — click to reveal</summary>
# MAGIC
# MAGIC - `feature_cols` is defined once in `TRAINING_CONFIG` and reused for the split and the importance chart — no risk of mismatch
# MAGIC - file writing is explicit at the end — easy to move into the Kedro catalog
# MAGIC - `pickle` is imported at the top, not mid-notebook
# MAGIC
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## your turn
# MAGIC
# MAGIC 📌 *this notebook runs, and the logic is readable. but it is still a notebook:*
# MAGIC *the execution order only exists in your head, and the paths and settings still live in Python.*
# MAGIC
# MAGIC *your job now is to move this into the Kedro project.*
# MAGIC *go back through the three sections and decide, for every piece of this notebook,*
# MAGIC *where it belongs in a Kedro project — and why.*
