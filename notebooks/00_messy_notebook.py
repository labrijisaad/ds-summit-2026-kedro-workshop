# Databricks notebook source
# MAGIC %md
# MAGIC <details>
# MAGIC <summary>How to read this notebook</summary>
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC 📌 *Every note marked with this pin was added by me — they are not part of the original notebook.*
# MAGIC
# MAGIC Each pin points at something in the cell just **above or below** it and asks you a question.
# MAGIC Stop for a few seconds and **answer it in your head** before moving on — that is where the learning happens.
# MAGIC
# MAGIC When there is an `answer — click to reveal` block, open it **only after** you have formed your own opinion.
# MAGIC
# MAGIC > 💡 The code itself is intentionally written the way real notebooks are written.
# MAGIC > It runs, and the results are fine. The problem is what happens when **someone else inherits it**.
# MAGIC
# MAGIC </details>
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC # Gym Churn - model v2 final FINAL
# MAGIC
# MAGIC predicting churn for the retention team
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC 📌 *you have been given this notebook by a colleague who just left for holiday.*
# MAGIC *he says the model is good and ready to ship.*
# MAGIC *your job today is to read it, run it, spot the problems, and eventually rewrite it properly.*
# MAGIC
# MAGIC *start by fixing the paths below so the notebook runs end to end.*

# COMMAND ----------

import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. load data
# MAGIC
# MAGIC 📌 **your turn — the notebook does not run yet.**
# MAGIC
# MAGIC *the two paths in the next cell point to the previous owner's personal folder.*
# MAGIC *that folder does not exist on your machine, so the cell fails with a* `FileNotFoundError`*.*
# MAGIC
# MAGIC *your data is in the* `data/01_raw/` *folder of this repository:*
# MAGIC `members.csv` *and* `checkins.parquet`*.*
# MAGIC *edit the two paths so the cell runs, then continue.*
# MAGIC
# MAGIC *try it yourself first — the answer is right after the cell if you get stuck.*

# COMMAND ----------

# TODO: fix these paths before running
members = pd.read_csv(
    "/Workspace/Users/firstname.lastname@axa.com/some/old/folder/members_final_v2.csv"
)
checkins = pd.read_parquet(
    "/Workspace/Users/firstname.lastname@axa.com/another/folder/checkins_export_jan.parquet"
)

print(members.shape)
print(checkins.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC <details>
# MAGIC <summary>answer — click to reveal</summary>
# MAGIC
# MAGIC the notebook lives in `notebooks/`, so the data is one folder up:
# MAGIC
# MAGIC ```python
# MAGIC members  = pd.read_csv("../data/01_raw/members.csv")
# MAGIC checkins = pd.read_parquet("../data/01_raw/checkins.parquet")
# MAGIC ```
# MAGIC
# MAGIC you should get `(8640, 12)` for members and `(150062, 5)` for checkins.
# MAGIC
# MAGIC </details>
# MAGIC
# MAGIC ---
# MAGIC 📌 *you just edited two paths to make someone else's notebook run on your machine.*
# MAGIC *now imagine the data moves to a new folder next month, and this notebook has fifteen read/write calls*
# MAGIC *spread over eighty cells. how do you find them all? how do you know you found them all?*

# COMMAND ----------

members.head()

# COMMAND ----------

members.isna().sum()

# COMMAND ----------

members.describe()

# COMMAND ----------

checkins.head()

# COMMAND ----------

checkins["activity"].value_counts()

# COMMAND ----------

checkins["days_ago"].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. target
# MAGIC
# MAGIC a member is **churned** if they have no check-in in the last 30 days.

# COMMAND ----------

recent = checkins[checkins["days_ago"] <= 30]
active_ids = recent["member_id"].unique()

members["churned"] = (~members["member_id"].isin(active_ids)).astype(int)

print(members["churned"].value_counts())
print(f"churn rate: {members['churned'].mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC 📌 *the number `30` is the business definition of churn. it is buried inside a filter with no label.*
# MAGIC *what happens when the retention team says "let's use 45 days instead"?*
# MAGIC
# MAGIC <details>
# MAGIC <summary>answer — click to reveal</summary>
# MAGIC
# MAGIC you would have to search every cell manually. `30` also appears again further down in the checkin features.
# MAGIC miss one and your features no longer match your target definition.
# MAGIC this is a parameter — it should be defined once, at the top, with a name.
# MAGIC
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. checkin features

# COMMAND ----------

history = checkins[checkins["days_ago"] > 30]

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

members.head()

# COMMAND ----------

# MAGIC %md
# MAGIC 📌 *this block does two things at once — it aggregates, then it merges.*
# MAGIC *if the aggregation logic changes, how do you test just that part without rerunning everything above?*
# MAGIC
# MAGIC <details>
# MAGIC <summary>answer — click to reveal</summary>
# MAGIC
# MAGIC you cannot. you have to rerun the whole notebook from the top.
# MAGIC wrapping this in a function like `build_checkin_features(members, checkins)` means
# MAGIC you can call it independently, test it, and change it without touching anything else.
# MAGIC
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. cleaning

# COMMAND ----------

print(f"before: {len(members)}")
members = members.drop_duplicates()
print(f"after:  {len(members)}")

# COMMAND ----------

members["age"] = members["age"].fillna(members["age"].median())
members["distance_km"] = members["distance_km"].fillna(members["distance_km"].median())
members["contract_type"] = members["contract_type"].fillna("Monthly")

members["club"] = members["club"].str.strip().str.title()
members["signup_channel"] = members["signup_channel"].str.strip()

members.isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC 📌 *`"Monthly"` in line 3 is a business decision — if a member has no contract type, we assume Monthly.*
# MAGIC *nothing in the code signals that this was intentional.*
# MAGIC *how would a new team member know this was a choice and not just a random default?*
# MAGIC
# MAGIC <details>
# MAGIC <summary>answer — click to reveal</summary>
# MAGIC
# MAGIC they would not. they might change it, delete it, or copy it wrong.
# MAGIC fill values like this belong in a config — named, visible, and easy to find.
# MAGIC
# MAGIC </details>

# COMMAND ----------

import matplotlib.pyplot as plt  # importing here, right where the chart is

members["age"].hist(bins=30)
plt.title("age distribution")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 📌 *`matplotlib` is imported here, in the middle of the notebook, because this is the first chart.*
# MAGIC *the same thing happens again in sections 5, 6, 7, 8 and 10 — each new library appears*
# MAGIC *the moment it is first needed. it feels natural while you write. what does it cost the next reader?*
# MAGIC
# MAGIC <details>
# MAGIC <summary>answer — click to reveal</summary>
# MAGIC
# MAGIC **you cannot tell what this notebook depends on without reading every single cell.**
# MAGIC there is no one place that answers "what do I need to install to run this?".
# MAGIC
# MAGIC it also breaks partial execution. say a colleague only wants to redraw the ROC curve
# MAGIC in section 8. they open a fresh kernel, run section 8, and get:
# MAGIC
# MAGIC ```
# MAGIC NameError: name 'plt' is not defined
# MAGIC ```
# MAGIC
# MAGIC because `plt` was imported back in section 4, in a cell that has nothing to do with plotting.
# MAGIC now they have to hunt upwards through the notebook to find which cells they are secretly
# MAGIC depending on — and that dependency is written nowhere.
# MAGIC
# MAGIC imports belong at the top: one block, visible immediately, independent of execution order.
# MAGIC
# MAGIC </details>

# COMMAND ----------

members["club"].value_counts()

# COMMAND ----------

members["contract_type"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. encode

# COMMAND ----------

from sklearn.preprocessing import LabelEncoder  # another import mid-notebook

cat_cols = ["contract_type", "club", "signup_channel"]

le = LabelEncoder()
for col in cat_cols:
    members[col] = le.fit_transform(members[col].astype(str))

members[cat_cols].head()

# COMMAND ----------

# MAGIC %md
# MAGIC 📌 *`cat_cols` is defined here. the same column names appear again in `feature_cols` further down.*
# MAGIC *if you add a new categorical column, will you remember to update both places?*
# MAGIC
# MAGIC <details>
# MAGIC <summary>answer — click to reveal</summary>
# MAGIC
# MAGIC probably not. miss one and the model silently trains without the new column.
# MAGIC a single `CAT_COLS = [...]` defined once at the top removes this risk entirely.
# MAGIC
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. split + save test set

# COMMAND ----------

from sklearn.model_selection import train_test_split  # another import mid-notebook

feature_cols = [
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
]

X = members[feature_cols]
y = members["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# save test for evaluation later
test_set = X_test.copy()
test_set["churned"] = y_test.values
test_set.to_csv(
    "../data/01_raw/test_set_DO_NOT_DELETE.csv",
    index=False,
)

print(f"train: {X_train.shape}")
print(f"test:  {X_test.shape}")
print(f"churn rate train: {y_train.mean():.1%}")
print(f"churn rate test:  {y_test.mean():.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC 📌 *the test set is saved here, inside the training section.*
# MAGIC *what happens if someone runs the evaluation cells without running this one first?*
# MAGIC
# MAGIC <details>
# MAGIC <summary>answer — click to reveal</summary>
# MAGIC
# MAGIC they silently evaluate on stale data from a previous run — or worse, on the wrong model entirely.
# MAGIC the dependency between the two sections is completely invisible.
# MAGIC this is one of the most common sources of silent errors in notebook-based workflows.
# MAGIC
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. train

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier  # another import mid-notebook

# tried 100 trees first, 200 seems better
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

model.fit(X_train, y_train)
print("done")

# COMMAND ----------

# MAGIC %md
# MAGIC 📌 *`n_estimators=200`, `max_depth=10`, `random_state=42` — all hardcoded.*
# MAGIC *are these the same values that produced the model currently in production?*
# MAGIC
# MAGIC <details>
# MAGIC <summary>answer — click to reveal</summary>
# MAGIC
# MAGIC there is no way to know. to run an experiment with different values you edit the code directly.
# MAGIC to reproduce a previous run you have to remember what you typed.
# MAGIC these are parameters — they belong in a config, not scattered in cells.
# MAGIC
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. results

# COMMAND ----------

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

y_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_proba)
print(f"AUC: {auc:.4f}")

# COMMAND ----------

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
plt.show()

# COMMAND ----------

fpr, tpr, _ = roc_curve(y_test, y_proba)

plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. feature importance

# COMMAND ----------

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
    ascending=False
)
print(importances)

# COMMAND ----------

importances.plot(kind="bar", figsize=(13, 5), title="Feature Importance")
plt.ylabel("importance")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 📌 *feature importance is useful — but in notebook form, this output is easy to lose.*
# MAGIC *where is this chart saved? how do you compare it to last week's model run?*
# MAGIC
# MAGIC <details>
# MAGIC <summary>answer — click to reveal</summary>
# MAGIC
# MAGIC nowhere by default. it appears on screen, then disappears.
# MAGIC if you want traceability, charts and metrics must be written to named outputs,
# MAGIC not just shown in a transient notebook session.
# MAGIC
# MAGIC </details>

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. save model

# COMMAND ----------

import pickle  # another import mid-notebook

# v2 - updated after retrain
with open("churn_model_v2_final.pkl", "wb") as f:
    pickle.dump(model, f)

print("saved")

# COMMAND ----------

# MAGIC %md
# MAGIC 📌 *every time this cell runs, `churn_model_v2_final.pkl` is overwritten.*
# MAGIC *which model is currently running in production? when was it trained? with what parameters?*
# MAGIC
# MAGIC <details>
# MAGIC <summary>answer — click to reveal</summary>
# MAGIC
# MAGIC nobody knows. there is no versioning, no timestamp, no record of the parameters used.
# MAGIC if you retrain tomorrow with different values, today's model is gone forever.
# MAGIC
# MAGIC </details>
