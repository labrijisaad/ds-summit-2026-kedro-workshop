"""
Section 1 of notebooks/01_clean_notebook - data processing.

The functions are copied as-is from the notebook. The only difference: they
no longer read or write files themselves - the Data Catalog does that.
"""

from sklearn.preprocessing import LabelEncoder


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
