<hr style="border-width:2px;border-color:darkred">
<div align="center">

![da_logo_small](_assets/da_logo_transparent_small.gif)

<h1>From Notebooks to Production with Kedro</h1>
<h3>Hands-on Workshop · Data Science Summit 2026, Amsterdam</h3>

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/)

**Saad Labriji** — AXA Direct Assurance

<br>
<hr style="border-width:2px;border-color:darkred">
</div>

## Overview

You inherit a gym churn model written in a messy notebook. Your job: turn it into a Kedro project that runs with one command, keeps paths and settings in config, and shows its whole flow as a graph.

**Kedro** is an open source Python framework for reproducible, maintainable data and ML pipelines. At AXA Direct Assurance, a dozen models run on Kedro in Databricks, with automated retraining and versioned models in production.

### Workshop Flow

```
Step 1 — Get familiar with Kedro (optional)
         Run Kedro's Spaceflights example: catalog, parameters, pipelines, Kedro-Viz.
   ↓
Step 2 — Run the messy notebook
         The churn model you inherited. Make it run, spot what would break in production.
   ↓
Step 3 — Read the clean notebook
         Provided to you: the same code split into functions, same results.
   ↓
Step 4 — Turn it into a Kedro project
         Fill the catalog, the parameters and the three pipelines.
   ↓
Done   — kedro run gives the notebook's AUC, kedro viz run shows the full graph.
```

> Already in Step 4? Jump to [Quick Reference](#quick-reference)

## Setup

Do this **before the session**. If something fails, ask before the workshop starts.

<details>
<summary><b>Install uv, Git and VS Code</b></summary>

### uv

[uv](https://docs.astral.sh/uv/) installs Python (3.11–3.13) and the dependencies — no separate Python install needed.

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

macOS / Linux:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Other methods — https://docs.astral.sh/uv/getting-started/installation/

Reopen your terminal, then check with `uv --version`.

### Git

https://git-scm.com/downloads

### VS Code

https://code.visualstudio.com/download — with the **Python** and **Jupyter** extensions.

</details>

```sh
git clone https://github.com/labrijisaad/ds-summit-2026-kedro-workshop.git
cd ds-summit-2026-kedro-workshop
uv sync --group dev

# activate — macOS / Linux
source .venv/bin/activate
# activate — Windows
.venv\Scripts\activate

kedro info      # prints the Kedro logo
```

Windows "running scripts is disabled": run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation and prefix commands with `uv run`.

## The Use Case — Gym Churn

A Paris gym chain wants to find the members who are about to leave.

- **Target** — `churned = 1` if the member has no check-in in the last 30 days. It is computed from the check-ins; there is no label column.
- **Features** — built only from check-ins older than 30 days (days 31–90), so the target does not leak into them.
- **Model** — random forest, scored with AUC on a 20% test set.

### The Data

Synthetic data in `data/01_raw/`. About one member in three churns.

**`members.csv`** — one row per member (8,000 members)

| Column | Description |
|---|---|
| `member_id` | Member identifier, e.g. `M00042` |
| `age` | Age in years |
| `tenure_months` | Months since the member joined |
| `monthly_fee_eur` | Monthly subscription fee (29.90 to 59.90 €) |
| `distance_km` | Distance from home to the club |
| `baseline_visits_pm` | Usual number of visits per month |
| `has_personal_trainer` | Has a personal trainer (`True` / `False`) |
| `joined_with_friend` | Signed up together with a friend (`True` / `False`) |
| `group_classes_pm` | Group classes attended per month |
| `contract_type` | `Monthly`, `Annual` or `Student` |
| `club` | Home club: `Paris-Nord`, `Paris-Sud`, `Paris-Est`, `Paris-Ouest`, `Paris-Centre` |
| `signup_channel` | `Website`, `Walk-in`, `Referral` or `Corporate` |

**`checkins.parquet`** — one row per visit over the last 90 days (~150,000 visits)

| Column | Description |
|---|---|
| `checkin_id` | Visit identifier |
| `member_id` | Who came — joins to `members.csv` |
| `days_ago` | How many days ago the visit happened (1 = yesterday, 90 = three months ago) |
| `duration_minutes` | Length of the visit |
| `activity` | `Cardio`, `Weights`, `Class` or `Swim` |

## Step 1 — Get familiar with Kedro *(optional)*

Kedro's Spaceflights example, shown in the presentation. Create it **outside** this repository:

```sh
cd ..
kedro new --starter spaceflights-pandas --name spaceflights
cd spaceflights
uv sync --group dev
uv run kedro run
uv run kedro viz run
```

Look at `conf/base/catalog.yml`, `conf/base/parameters.yml`, and `nodes.py` / `pipeline.py` in `src/spaceflights/pipelines/`. Then go back with `cd ../gym-churn-workshop`.

## Step 2 — Run the messy notebook

The notebook you inherited. Open it in one of two ways:

- **VS Code** — open `notebooks/00_messy_notebook.ipynb`, click **Select Kernel** → `.venv`
- **JupyterLab** — run `uv run jupyter lab` in your terminal, then open `notebooks/00_messy_notebook.ipynb`

1. Run it top to bottom. The data-loading cell fails — fix it.
2. Continue until the model trains and the charts appear.
3. Answer the 📌 questions before opening the reveals.

The `.py` files next to the notebooks are the same notebooks in Databricks format — you can ignore them.

## Step 3 — Read the clean notebook

Open `notebooks/01_clean_notebook.ipynb`. It is provided to you: the messy code refactored into functions, with paths and settings at the top, and the same results. These functions become your Kedro nodes.

## Step 4 — Turn it into a Kedro project

**Done when** `kedro run` prints **AUC ≈ 0.6512** and `kedro viz run` shows the three pipelines connected.

| Pipeline | What it does |
|---|---|
| `data_processing` | builds the target and the features, cleans and encodes |
| `model_training` | splits the data and trains the model |
| `model_evaluation` | computes the AUC and produces the charts |

The functions are already in `src/gym_churn/pipelines/<pipeline>/nodes.py`, and each `pipeline.py` has TODO hints. You need to:

1. Declare the inputs and outputs in `conf/base/catalog.yml`.
2. Move the config dictionaries into `conf/base/parameters.yml`.
3. Wire the nodes in each `pipeline.py`.

```sh
kedro run                              # everything
kedro run --pipeline=data_processing   # one pipeline
kedro viz run                          # graph at http://127.0.0.1:4141/
```

Until nodes are wired, `kedro run` fails with `Pipeline contains no nodes`. Once `metrics` is in the catalog, the AUC is written to `data/08_reporting/metrics.json`.

## Quick Reference

**Catalog** — `conf/base/catalog.yml`

```yaml
members:
  type: pandas.CSVDataset
  filepath: data/01_raw/members.csv

checkins:
  type: pandas.ParquetDataset
  filepath: data/01_raw/checkins.parquet
```

**Parameters** — `conf/base/parameters.yml`

```yaml
churn_window_days: 30
n_estimators: 200
```

Used in a node as `"params:churn_window_days"`. Nested: `"params:training.n_estimators"`; `"params:training"` passes the whole dict.

**A node** — `pipeline.py`

```python
from kedro.pipeline import Node, Pipeline

Node(
    func=build_target,
    inputs=["members", "checkins", "params:churn_window_days"],
    outputs="members_with_target",
    name="build_target_node",
)
```

Several return values → a list of outputs: `outputs=["X_train", "X_test", "y_train", "y_test"]`.

**Dataset types** — [reference](https://docs.kedro.org/projects/kedro-datasets/en/latest/)

<details>
<summary><b>Hint — which type for which output</b></summary>

| What | `type` |
|---|---|
| DataFrame as CSV | `pandas.CSVDataset` (add `save_args: {index: False}` to drop the index) |
| DataFrame as Parquet | `pandas.ParquetDataset` |
| Any Python object (model, Series) | `pickle.PickleDataset` |
| Dictionary as JSON | `json.JSONDataset` |
| Matplotlib figure as PNG | `matplotlib.MatplotlibDataset` |

</details>

## Troubleshooting

| Message | What it means |
|---|---|
| `Pipeline contains no nodes` | Nothing wired yet in `pipeline.py`. |
| `Pipeline input(s) {'...'} not found in the DataCatalog` | Typo in an input name, `params:` key missing from `parameters.yml`, or you ran one pipeline alone and its inputs were never saved to disk. |
| `Output(s) [...] are returned by more than one nodes` | Two nodes write the same output name. |
| `Failed to save outputs of node ...` | Number of `outputs` ≠ number of values the function returns. |
| `kedro: command not found` / `not recognized` | Environment not activated — activate it or use `uv run kedro ...`. |
| A Series comes back as a DataFrame | It was saved as CSV — use `pickle.PickleDataset` or keep it in memory. |

`register_new_resolver` warnings can be ignored. To turn off Kedro telemetry, set `DO_NOT_TRACK=1`.

## Going Further

<details>
<summary><b>Click to expand</b></summary>

- **Versioning** — add `versioned: true` to the model in `catalog.yml`, run twice, look at `data/06_models/`.
- **Metrics history** — version `metrics` too, run with different `--params`, compare the files.
- **Namespaces** — train the same pipeline with two sets of hyperparameters.

</details>

## Documentation

- Kedro concepts — https://docs.kedro.org/en/stable/getting-started/kedro_concepts/
- Nodes — https://docs.kedro.org/en/stable/build/nodes/
- Pipelines — https://docs.kedro.org/en/stable/build/pipeline_introduction/
- Parameters — https://docs.kedro.org/en/stable/configure/parameters/
- Catalog examples — https://docs.kedro.org/en/stable/catalog-data/data_catalog_yaml_examples/
- Dataset types — https://docs.kedro.org/projects/kedro-datasets/en/latest/
- Namespaces — https://docs.kedro.org/en/stable/build/namespaces/
- Kedro-Viz — https://docs.kedro.org/projects/kedro-viz/en/latest/

## Solution

The solution repo is at **[github.com/labrijisaad/ds-summit-2026-kedro-workshop-solutions](https://github.com/labrijisaad/ds-summit-2026-kedro-workshop-solutions)** — it is private for now and will be made public **15 minutes before the end of the workshop**.

<br>
<hr style="border-width:2px;border-color:darkred">
