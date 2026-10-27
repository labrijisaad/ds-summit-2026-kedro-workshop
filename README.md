<hr style="border-width:2px;border-color:darkred">
<div align="center">

![da_logo_small](_assets/da_logo_transparent_small.gif)

<h1>From Notebooks to Production with Kedro</h1>
<h3>Hands-on Workshop &middot; Data Science Summit 2026 &middot; Amsterdam &middot; 4 November</h3>

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://docs.astral.sh/uv/)

**Saad Labriji** - ML / GenAI Engineer at AXA Direct Assurance (France)

<br>
<hr style="border-width:2px;border-color:darkred">
</div>

## Overview

You inherit a gym churn model written in a messy notebook. Your job: turn it into a Kedro project that **runs with one command**, keeps every path and setting in **config**, and shows its whole flow as a **graph**.

**Kedro** is an open source Python framework for reproducible, maintainable data and ML pipelines. At AXA Direct Assurance, a dozen models run on Kedro in Databricks, with automated retraining and versioned models in production.

## Workshop Flow

Four steps - click to jump.

| | Step | What you do | Time |
|---|---|---|---|
| 0 | **[Setup](#setup)** | Install `uv`, clone the repo, `uv sync` - **before the session** | before |
| 1 | **[Get familiar with Kedro](#step-1---get-familiar-with-kedro-optional)** *(optional)* | Explore a ready-made Kedro project - shown live in the presentation | ~10 min |
| 2 | **[Run the messy notebook](#step-2---run-the-messy-notebook)** | The model you inherited. Make it run, spot what would break in production | ~10 min |
| 3 | **[Read the clean notebook](#step-3---read-the-clean-notebook)** | Same code, split into functions. These become your Kedro nodes | ~5 min |
| 4 | **[Turn it into a Kedro project](#step-4---turn-it-into-a-kedro-project)** | Fill the catalog, the parameters and the three pipelines | ~30 min |

**Done when** `kedro run` reproduces the notebook's AUC and `kedro viz run` shows the full graph.

Also here: **[Quick reference](#quick-reference)** &middot; **[Going further](#going-further)** &middot; **[Documentation](#documentation)** &middot; **[Solution](#solution)**

## Setup

Do this **before the session**. If something fails, ask before the workshop starts.

<details>
<summary><b>Install uv, Git and VS Code</b></summary>

### uv

[uv](https://docs.astral.sh/uv/) installs Python (3.11 to 3.13) and all dependencies for you - no separate Python installation needed.

Follow the one-line installer for your OS here:
**https://docs.astral.sh/uv/getting-started/installation/**

Then reopen your terminal and check:

```sh
uv --version
```

### Git

https://git-scm.com/downloads

### VS Code

https://code.visualstudio.com/download - with the **Python** and **Jupyter** extensions.

</details>

```sh
git clone https://github.com/labrijisaad/ds-summit-2026-kedro-workshop.git
cd ds-summit-2026-kedro-workshop
uv sync --group dev
```

Check that it worked:

```sh
uv run kedro info      # prints the Kedro logo
```

> Every command in this README is prefixed with `uv run`, so you never have to activate
> the environment yourself. If you prefer to activate it:
> `source .venv/bin/activate` (macOS / Linux) or `.venv\Scripts\activate` (Windows).

> **🛑 STOP** - do not start Step 1 until it is announced.

## Step 1 - Get familiar with Kedro *(optional)*

> **This step is done live during the presentation.** You don't have to run it yourself -
> watch, then come back to it later if you want to poke around on your own.

The goal here is **not to build anything**. It is to open an already-finished Kedro project, run it, and understand what is going on: where the data is declared, where the settings live, how the steps connect, and what the graph looks like. Once that clicks, Step 4 is mostly typing.

We use Kedro's official **Spaceflights** example. Create it **outside** this repository:

```sh
cd ..
uvx kedro new --starter spaceflights-pandas --name spaceflights
cd spaceflights
uv sync --group dev
uv run kedro run
uv run kedro viz run
```

<details>
<summary><b>Hint - what to look at, in this order</b></summary>

| Look at | And ask yourself |
|---|---|
| The `kedro run` logs | Nobody wrote this execution order - where does it come from? |
| `conf/base/catalog.yml` | Every dataset has a **name**, a **type** and a **path**. Notice there is no `read_csv` anywhere in the code. |
| `conf/base/parameters.yml` | Every setting, named and in one place. No magic numbers in the code. |
| `src/spaceflights/pipelines/*/nodes.py` | Plain Python functions. No Kedro imports, no file I/O - they could be unit-tested as-is. |
| `src/spaceflights/pipelines/*/pipeline.py` | Each function wrapped in a `Node` with named `inputs` and `outputs`. **This is the only place the order is implied.** |
| Kedro-Viz at http://127.0.0.1:4141/ | Click a node to see its code. Click a dataset to see its type and path. |

The key idea: the **output name of one node is the input name of the next**. That is it - that is how Kedro builds the graph.

</details>

> **🛑 STOP** - do not start Step 2 until it is announced.

## Step 2 - Run the messy notebook

Come back to the workshop repo:

```sh
cd ../ds-summit-2026-kedro-workshop
```

The notebook you inherited. Open it in one of two ways:

- **VS Code** - open `notebooks/00_messy_notebook.ipynb`, click **Select Kernel** and pick `.venv`
- **JupyterLab** - run `uv run jupyter lab`, then open `notebooks/00_messy_notebook.ipynb`

1. Run it top to bottom. The data-loading cell fails - fix it.
2. Continue until the model trains and the charts appear.
3. Answer the pinned questions **in your head first**, then open the reveals.

### The use case - Gym Churn

A Paris gym chain wants to find the members who are about to leave.

- **Target** - `churned = 1` if the member has no check-in in the last 30 days. It is computed from the check-ins; there is no label column in the data.
- **Features** - built **only** from check-ins older than 30 days (days 31 to 90), so the target never leaks into the inputs.
- **Model** - random forest, scored with AUC on a 20% test set.

> **Don't focus on the ML.** The point of today is refactoring, not modelling. Do not try to improve the AUC - reproduce it with a clean structure.

<details>
<summary><b>The data - quick look</b></summary>

Synthetic data in `data/01_raw/`. About one member in three churns.

**`members.csv`** - one row per member (~8,000 members, plus some duplicates)

| Column | Description |
|---|---|
| `member_id` | Member identifier, e.g. `M00042` |
| `age` | Age in years |
| `tenure_months` | Months since the member joined |
| `monthly_fee_eur` | Monthly subscription fee (29.90 to 59.90 EUR) |
| `distance_km` | Distance from home to the club |
| `baseline_visits_pm` | Usual number of visits per month |
| `has_personal_trainer` | Has a personal trainer (`True` / `False`) |
| `joined_with_friend` | Signed up together with a friend (`True` / `False`) |
| `group_classes_pm` | Group classes attended per month |
| `contract_type` | `Monthly`, `Annual` or `Student` |
| `club` | Home club: `Paris-Nord`, `Paris-Sud`, `Paris-Est`, `Paris-Ouest`, `Paris-Centre` |
| `signup_channel` | `Website`, `Walk-in`, `Referral` or `Corporate` |

**`checkins.parquet`** - one row per visit over the last 90 days (~150,000 visits)

| Column | Description |
|---|---|
| `checkin_id` | Visit identifier |
| `member_id` | Who came - joins to `members.csv` |
| `days_ago` | How many days ago the visit happened (1 = yesterday, 90 = three months ago) |
| `duration_minutes` | Length of the visit |
| `activity` | `Cardio`, `Weights`, `Class` or `Swim` |

</details>

> **🛑 STOP** - do not start Step 3 until it is announced.

## Step 3 - Read the clean notebook

Open `notebooks/01_clean_notebook.ipynb`. This one is given to you: the same code refactored into functions, with every path and setting declared at the top, and exactly the same results.

**These functions are your Kedro nodes** - they are already copied into `src/gym_churn/pipelines/<pipeline>/nodes.py`.

> **🛑 STOP** - do not start Step 4 until it is announced.

## Step 4 - Turn it into a Kedro project

This repository is your starting point. The node functions are already written - your job is the **configuration and the wiring**, so the project ends up looking like the Kedro-Viz graph shown in the presentation.

| Pipeline | What it does |
|---|---|
| `data_processing` | builds the target and the check-in features, cleans and encodes |
| `model_training` | splits the data and trains the model |
| `model_evaluation` | computes the AUC and produces the charts |

Three things to do:

1. **Declare the datasets** you want to keep in `conf/base/catalog.yml`
   *(anything not declared stays in memory and disappears at the end of the run)*
2. **Move `PROCESSING_CONFIG` and `TRAINING_CONFIG`** from the clean notebook into
   `conf/base/parameters.yml`
3. **Wire one `Node` per function** in each `pipeline.py` - every file has TODO hints

```sh
uv run kedro run                               # everything
uv run kedro run --pipeline=data_processing    # one pipeline
uv run kedro viz run                           # graph at http://127.0.0.1:4141/
```

**Done when** `kedro run` completes, `data/08_reporting/metrics.json` contains **AUC = 0.6512**, and `kedro viz run` shows the three pipelines connected end to end.

<details>
<summary><b>Hint - suggested order of attack</b></summary>

Don't try to wire everything and then run once. Build it one pipeline at a time:

1. Add the parameters for `data_processing`, wire its 4 nodes, run
   `uv run kedro run --pipeline=data_processing`. Declare `model_input_table` in the
   catalog so the next pipeline has something to read.
2. Add the training parameters, wire `model_training`, run it. Declare the model, and the
   test split you will need for evaluation.
3. Wire `model_evaluation`, declare `metrics` and the three plots, then
   `uv run kedro run` for the whole thing.

After each step, `uv run kedro viz run` and check the graph grew the way you expected.

</details>

<details>
<summary><b>Hint - two things that will trip you up</b></summary>

- **Running one pipeline alone fails** with `Pipeline input(s) {'...'} not found in the DataCatalog`. That dataset was never written to disk - either declare it in the catalog, or run the whole project with `uv run kedro run`.
- **`Failed to save outputs of node ...`** means the number of names in `outputs=` does not match the number of values the function returns. **Read the signature in `nodes.py` before wiring** - not every function returns a single object.

</details>

## Quick Reference

**Catalog** - `conf/base/catalog.yml`

```yaml
members:
  type: pandas.CSVDataset
  filepath: data/01_raw/members.csv

checkins:
  type: pandas.ParquetDataset
  filepath: data/01_raw/checkins.parquet
```

**Parameters** - `conf/base/parameters.yml`

```yaml
churn_window_days: 30
n_estimators: 200
```

Used in a node as `"params:churn_window_days"`.
Nested keys: `"params:training.n_estimators"`, or `"params:training"` to pass the whole dict.
Override at run time without touching the code:

```sh
uv run kedro run --params "n_estimators=50,max_depth=5"
```

**A node** - `pipeline.py`

```python
from kedro.pipeline import Node, Pipeline

Node(
    func=build_target,
    inputs=["members", "checkins", "params:churn_window_days"],
    outputs="members_with_target",
    name="build_target_node",
)
```

The output name of one node is the input name of the next - that is how Kedro works out the execution order. Several return values means a list of outputs:

```python
outputs = ["X_train", "X_test", "y_train", "y_test"]
```

<details>
<summary><b>Hint - which dataset type for which output</b></summary>

| What | `type` |
|---|---|
| DataFrame as CSV | `pandas.CSVDataset` (add `save_args: {index: False}` to drop the index) |
| DataFrame as Parquet | `pandas.ParquetDataset` |
| Any Python object (model, Series) | `pickle.PickleDataset` |
| Dictionary as JSON | `json.JSONDataset` |
| Matplotlib figure as PNG | `matplotlib.MatplotlibDataset` |

Full list - https://docs.kedro.org/projects/kedro-datasets/en/latest/

</details>

<details>
<summary><b>Hint - useful commands while you work</b></summary>

```sh
uv run kedro registry list              # which pipelines Kedro can see
uv run kedro catalog list               # which datasets Kedro can see
uv run kedro run --nodes <node_name>    # run a single node
uv run kedro viz run                    # the graph - your best debugging tool
uv run kedro jupyter lab                # notebook with catalog, context, pipelines preloaded
```

In a `kedro jupyter lab` session you can inspect any declared dataset directly:

```python
catalog.load("model_input_table").head()
```

</details>

## Going Further

Finished early? Pick one.

<details>
<summary><b>Click to expand</b></summary>

- **Easy - Versioning.** Add `versioned: true` to the model in `catalog.yml`, run twice, and look at what happened in `data/06_models/`. Which problem from the messy notebook does this fix?
- **Easy - Metrics history.** Version `metrics` as well, run a few times with different `--params`, then compare the files. You now have a record of which settings produced which score.
- **Medium - Namespaces.** Train the same pipeline twice with two different sets of hyperparameters, and see how it changes the Kedro-Viz graph. Docs - https://docs.kedro.org/en/stable/build/namespaces/
- **Hard - Deploy Kedro-Viz.** `kedro viz build` produces a static site in `build/` that you can host anywhere: GitHub Pages, Azure Static Web Apps, AWS S3 + CloudFront, GCP Cloud Storage. The **solution repository contains a working example**: the graph is built and published to an **Azure Static Web App**, fully automated with a **GitHub Actions CI/CD workflow** that redeploys on every push to `main`. Use it as a template and deploy yours wherever you like.

</details>

## Documentation

Kedro's docs are short, well written and full of examples - a quick look will get you through the wiring much faster than guessing.

- Kedro concepts - https://docs.kedro.org/en/stable/getting-started/kedro_concepts/
- Nodes - https://docs.kedro.org/en/stable/build/nodes/
- Pipelines - https://docs.kedro.org/en/stable/build/pipeline_introduction/
- Parameters - https://docs.kedro.org/en/stable/configure/parameters/
- Catalog examples - https://docs.kedro.org/en/stable/catalog-data/data_catalog_yaml_examples/
- Dataset types - https://docs.kedro.org/projects/kedro-datasets/en/latest/
- Namespaces - https://docs.kedro.org/en/stable/build/namespaces/
- Kedro-Viz - https://docs.kedro.org/projects/kedro-viz/en/latest/

## Solution

The full solution - including the deployed Kedro-Viz graph and its CI/CD workflow - lives at
**[github.com/labrijisaad/ds-summit-2026-kedro-workshop-solutions](https://github.com/labrijisaad/ds-summit-2026-kedro-workshop-solutions)**.

It is private for now and will be made public **about 30 minutes before the end of the workshop** - try it yourself first.

## Author

**Saad Labriji** - AXA Direct Assurance

- GitHub - [github.com/labrijisaad](https://github.com/labrijisaad)
- LinkedIn - [linkedin.com/in/labrijisaad](https://www.linkedin.com/in/labrijisaad/)

<br>
<hr style="border-width:2px;border-color:darkred">
<div align="center">

**Thanks for coming - and congratulations, you just shipped a prod-ready ML pipeline.**

</div>
