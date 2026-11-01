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

You inherit a gym churn model written in a messy notebook. Your job: turn it into a Kedro
project that **runs with one command**, keeps every path and setting in **config**, and
shows its whole flow as a **graph** (Kedro-Viz).

**Kedro** is an open source Python framework for reproducible, maintainable data and ML
pipelines. At AXA Direct Assurance, a dozen models run on Kedro in Databricks, with
automated retraining in production.

## Workshop Flow

Four steps - click to jump.

| | Step | What you do |
|---|---|---|
| 0 | Setup | Install `uv`, clone the repo, run `uv sync` - **before the session** |
| 1 | Get familiar with Kedro *(optional)* | Explore a ready-made Kedro project - shown live during the presentation |
| 2 | Run the messy notebook | The model you inherited. Make it run, and spot what would break in production |
| 3 | Read the clean notebook | The same code, split into functions. These become your Kedro nodes. Normally you would do this refactoring yourself, but today we want to spend the time on Kedro |
| 4 | Turn it into a Kedro project | Fill in the catalog, the parameters and the three pipelines, then open Kedro-Viz |

**The main workshop is done when** `kedro run` reproduces the notebook's AUC and
`kedro viz run` shows the full graph. If you finish early, there are extra challenges in
[Going further](#going-further).

Also useful: **[Going further](#going-further)** &middot; **[Documentation](#documentation)** &middot; **[Solution](#solution)**

## Setup

Do this **before the session** (or at the very beginning, during the presentation).
If something fails, please ask before the workshop starts.

<details>
<summary><b>Install uv, Git and VS Code</b></summary>

### uv

[uv](https://docs.astral.sh/uv/) installs Python (3.11 to 3.13) and all dependencies for
you - no separate Python installation needed.

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
> watch now, and come back to it later if you want to explore on your own.

The goal here is **not to build anything**. It is to open an already-finished Kedro
project, run it, and understand what is going on: where the data is declared, where the
settings live, how the steps connect, and what the graph looks like. Once that clicks,
Step 4 is mostly typing.

We use Kedro's official **Spaceflights** example. Create it **outside** this repository:

```sh
cd ..
uvx kedro new --starter spaceflights-pandas --name spaceflights
cd spaceflights
uv sync
uv run kedro run        # run the project yourself and train the model locally
uv run kedro viz run    # open Kedro-Viz to see the different components of the project
```

<details>
<summary><b>Hint - what to look at, in this order</b></summary>

| Look at | And ask yourself |
|---|---|
| `conf/base/catalog.yml` | Every dataset has a **name**, a **type** and a **path**. Notice there is no `read_csv` anywhere in the code. |
| `conf/base/parameters.yml` | Every setting, named and in one place. No magic numbers in the code. |
| `src/spaceflights/pipelines/*/nodes.py` | Plain Python functions. No Kedro imports, no file I/O - they could be unit-tested as-is. |
| `src/spaceflights/pipelines/*/pipeline.py` | Each function wrapped in a `Node` with named `inputs` and `outputs`. **This is the only place the order is implied.** |
| The `kedro run` logs | Nobody wrote this execution order - where does it come from? |
| Kedro-Viz at http://127.0.0.1:4141/ | Click a node to see its code. Click a dataset to see its type and path. |

The key idea: the **output name of one node is the input name of the next**. That is it -
that is how Kedro builds the graph.

</details>

> **🛑 STOP** - do not start Step 2 until it is announced.

## Step 2 - Run the messy notebook

Come back to the workshop repo:

```sh
cd ../ds-summit-2026-kedro-workshop
```

This is our use case for today: a gym churn classification problem.

> **Don't focus on the ML or the data.** The goal is to practise Kedro, not to change the
> modelling code.

### The use case - Gym Churn

A Paris gym chain wants to find the members who are about to leave.

- **Target** - `churned = 1` if the member has no check-in in the last 30 days. It is
  computed from the check-ins; there is no label column in the data.
- **Features** - built **only** from check-ins older than 30 days (days 31 to 90), so the
  target never leaks into the inputs.
- **Model** - random forest, scored with AUC on a 20% test set.

> Do not try to improve the AUC - reproduce it with a clean structure.

Open the notebook you inherited in one of two ways:

- **VS Code** - open `notebooks/00_messy_notebook.ipynb`, click **Select Kernel** and pick `.venv`
- **JupyterLab** - run `uv run jupyter lab`, then open `notebooks/00_messy_notebook.ipynb`

1. Run it from top to bottom. The data-loading cell fails - fix it (there is a hint in the
   notebook). As you go, read the code and spot the problems for yourself.
2. Continue until the model trains and the charts appear.
3. Answer the pinned questions **in your head first**, then open the reveals.

<details>
<summary><b>The data - quick look</b></summary>

This is synthetic data created for this workshop, in `data/01_raw/`. About one member in
three churns.

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

Open `notebooks/01_clean_notebook.ipynb`. This one is given to you: the same code
refactored into functions, with every path and setting declared at the top, and exactly
the same results. Normally you would do this refactoring yourself - today we skip it to
spend the time on Kedro.

**These functions are your Kedro nodes.** Keep this notebook open during Step 4 - you will
copy the function bodies out of it:

| Notebook section | Goes into |
|---|---|
| 1. data processing | `data_processing/nodes.py` - **already done for you** |
| 2. model training | `model_training/nodes.py` - signatures given, bodies are yours |
| 3. model evaluation | `model_evaluation/nodes.py` - signatures given, bodies are yours |

Also note `INPUTS` / `OUTPUTS` / `PROCESSING_CONFIG` / `TRAINING_CONFIG` at the top - every
one of those entries has a home in `conf/base/`.

> **🛑 STOP** - do not start Step 4 until it is announced.

## Step 4 - Turn it into a Kedro project (your turn)

### Your target

The finished graph is live here - **this is your spec**:

> ### **[→ Open the target Kedro-Viz graph  ←](https://zealous-mushroom-0953eca03.2.azurestaticapps.net/?pid=__default__&expandAllPipelines=false&types=nodes,datasets)**

Keep it open in a second tab. Read off it:

- **node names** and in which pipeline they live
- **which dataset feeds which node** - that is exactly your `inputs=` / `outputs=`
- **which parameters** each node consumes (`params:...`)
- which intermediate datasets exist at all

> **The grey boxes are namespaces - ignore them.** A flat `Pipeline([...])` gives you the
> same nodes, the same AUC and the same wiring, just without the boxes. Namespaces are the
> bonus exercise in [Going further](#going-further).

### What to build

Three pipelines, one worked example, two to build.

| Pipeline | Status | What to do |
|---|---|---|
| `data_processing` | Nodes written, 1/4 wired | Add the 3 remaining `Node(...)` blocks in `pipeline.py` by copying the pattern of the first one |
| `model_training` | Function **signatures** only | Copy the bodies from `notebooks/01_clean_notebook` section 2, then wire the nodes |
| `model_evaluation` | Function **signatures** only | Same, section 3 |

Plus two config files to finish:

1. `conf/base/catalog.yml` - two datasets are declared, add the rest you want to persist
2. `conf/base/parameters.yml` - one parameter is declared, add the rest from the notebook

> **The stub `nodes.py` files have no imports** - only the signatures. When you paste a
> body in, add the import at the **top of the file**. That is the fix for one of the
> problems you found in the messy notebook.

```sh
uv run kedro run --pipeline=data_processing   # test one pipeline at a time
uv run kedro run                              # everything
uv run kedro viz run                          # graph at http://127.0.0.1:4141/
```

**Done when** `uv run kedro run` completes, `data/08_reporting/metrics.json` shows
**`"auc": 0.6512...`** (first four decimals), and `kedro viz run` shows the three
pipelines connected end to end.

> Until you wire them, `model_training` and `model_evaluation` are empty and their
> nodes raise `NotImplementedError`. Both are expected - that is your TODO list,
> not a broken starter.

### The payoff - run this once it is green

```sh
uv run kedro run --params "churn_window_days=45"
```

One number, one command, every node downstream recomputed consistently - target *and*
features. In the messy notebook that `30` was hardcoded in two different cells. **This is
the whole point of the workshop.**

<details>
<summary><b>Hint - the two things that will trip you up</b></summary>

- **A function returning several values needs a list of outputs.** `split_data` returns
  4 values, so `outputs=["X_train", "X_test", "y_train", "y_test"]`. `evaluate_model`
  returns 3. If the count does not match you get `Failed to save outputs of node ...`.
- **Running one pipeline alone fails** with `Pipeline input(s) {'...'} not found in the
  DataCatalog`. That dataset was never written to disk - either declare it in the
  catalog, or run the whole project.

</details>

<details>
<summary><b>Hint - which dataset type for which output</b></summary>

| What | `type` |
|---|---|
| DataFrame as Parquet | `pandas.ParquetDataset` |
| DataFrame as CSV | `pandas.CSVDataset` (add `save_args: {index: False}`) |
| Any Python object (model, Series) | `pickle.PickleDataset` |
| Dict as JSON | `json.JSONDataset` |
| Matplotlib figure as PNG | `matplotlib.MatplotlibDataset` |

**Use these for the split** - don't improvise here:

| Dataset | Type | Why |
|---|---|---|
| `model_input_table` | `pandas.ParquetDataset` | keeps dtypes |
| `X_train` / `X_test` | `pandas.ParquetDataset` | keeps dtypes |
| `y_train` / `y_test` | `pickle.PickleDataset` | CSV would turn a Series into a DataFrame and shift your AUC |

Full list: https://docs.kedro.org/projects/kedro-datasets/en/latest/

</details>

Useful while you work:

```sh
uv run kedro registry list               # which pipelines Kedro can see
uv run kedro catalog describe-datasets   # which datasets Kedro can see
uv run kedro viz run                     # the graph - your best debugging tool
```

## Going Further

Finished early? Pick one.

- **Easy - Versioning.** Add `versioned: true` to the model in `catalog.yml`, run twice,
  look at `data/06_models/`. Which problem from the messy notebook does this fix?
- **Easy - Metrics history.** Version `metrics` too, run with different `--params`,
  compare the files.
- **Medium - Namespaces.** Wrap each pipeline in a `namespace="..."` to get the grey boxes
  from the target graph:

  ```python
  return Pipeline(
      [...],
      namespace="model_training",
      prefix_datasets_with_namespace=False,   # node names prefixed, dataset names untouched
  )
  ```

  Drop that last argument (the default is `True`) and every internal dataset and
  parameter gets prefixed too - `model_training.X_train`, `params:model_training.test_size`.
  Then your catalog keys must match **exactly**, and if they don't, Kedro does **not**
  error: it silently falls back to a `MemoryDataset` and writes nothing. The only clue is
  the dataset type in the run log. Worth breaking on purpose once.
  Docs: https://docs.kedro.org/en/stable/build/namespaces/
- **Hard - Deploy Kedro-Viz.** `kedro viz build` makes a static site in `build/` that
  you can host anywhere - that is exactly how the target graph above is published. The
  solution repo has a working GitHub Actions workflow that deploys to an Azure Static Web
  App on every push to `main`.

## Documentation

- Kedro concepts - https://docs.kedro.org/en/stable/getting-started/kedro_concepts/
- Nodes - https://docs.kedro.org/en/stable/build/nodes/
- Pipelines - https://docs.kedro.org/en/stable/build/pipeline_introduction/
- Parameters - https://docs.kedro.org/en/stable/configure/parameters/
- Catalog examples - https://docs.kedro.org/en/stable/catalog-data/data_catalog_yaml_examples/
- Dataset types - https://docs.kedro.org/projects/kedro-datasets/en/latest/
- Namespaces - https://docs.kedro.org/en/stable/build/namespaces/
- Kedro-Viz - https://docs.kedro.org/projects/kedro-viz/en/latest/

## Solution

The full solution lives at
**[github.com/labrijisaad/ds-summit-2026-kedro-workshop-solutions](https://github.com/labrijisaad/ds-summit-2026-kedro-workshop-solutions)**.
Public ~30 minutes before the end of the workshop - try it yourself first.

## Author

**Saad Labriji** - AXA Direct Assurance

- GitHub - [github.com/labrijisaad](https://github.com/labrijisaad)
- LinkedIn - [linkedin.com/in/labrijisaad](https://www.linkedin.com/in/labrijisaad/)

<br>
<hr style="border-width:2px;border-color:darkred">
<div align="center">

**Thanks for coming - and congratulations, you just shipped a prod-ready ML pipeline.**

</div>
