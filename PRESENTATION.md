# Presentation — From Notebooks to Production with Kedro

Saad Labriji · Data Science Summit 2026, Amsterdam · 1h30 · 14 slides

For each slide: **Slide** = bullets to put on it. **Say** = speaker notes.

| Time | Part | Slides |
|---|---|---|
| 0:00 – 0:10 | Intro and the problem | 1 – 3 |
| 0:10 – 0:15 | Use case and finish line | 4 – 5 |
| 0:15 – 0:30 | Kedro in 15 minutes (live) | 6 – 10 |
| 0:30 – 1:15 | Hands-on | 11 |
| 1:15 – 1:23 | Solution | 12 |
| 1:23 – 1:30 | AXA, recap, Q&A | 13 – 14 |

---

## 1 — Title

**Slide**
- From Notebooks to Production with Kedro
- Saad Labriji — Data Scientist, AXA Direct Assurance
- QR code → workshop repo

**Say**
- 90 minutes, mostly hands-on
- Clone the repo and run `uv sync` now, it installs while I talk
- 30 seconds about me: team, what I build, why this topic

---

## 2 — The notebook you inherit

**Slide**
- `model_v2_final_FINAL.ipynb` — "works on my machine", author on holiday
- Paths to someone's personal folder
- Magic numbers: `30`, `"Monthly"`, `200` buried in cells
- Execution order only in the author's head
- Imports scattered → `NameError` on partial runs
- Charts on screen only, model overwritten every run

**Say**
- Hands up: who has inherited a notebook? Who has handed one over?
- None of these is a bug — the notebook runs, the AUC is fine
- The cost comes later: change the churn window → hunt every `30`; which model is in production? nobody knows
- You'll run this exact notebook in a few minutes

---

## 3 — What we want instead: Kedro

**Slide**
- One command: `kedro run`
- Paths in one file, settings in one file
- Each step = a plain, testable Python function
- Flow visible as a graph
- Outputs saved, models versioned
- Kedro: open source, LF AI & Data Foundation, runs anywhere Python runs

**Say**
- Kedro structures the code; it is not an orchestrator or an experiment tracker
- Today: same model, same AUC, production-ready structure

---

## 4 — Use case: gym churn

**Slide**
- Paris gym chain: who is about to leave?
- **Churned** = no check-in in the last 30 days (32% of members)
- `members.csv` — 8,000 members (+640 duplicates, missing values, dirty strings)
- `checkins.parquet` — 150,062 visits over 90 days
- Features from days 31–90 only · target from days 1–30
- Random Forest, AUC **0.6512**

**Say**
- No label column: we build the target from the check-ins
- Leakage trap: the last 30 days define the target, so they can't be features → same `30` twice → must be **one** parameter
- Synthetic data, deliberately messy
- Don't try to improve the model today — reproduce 0.6512 with a clean structure

---

## 5 — The finish line

**Slide**
- Kedro-Viz screenshot of the solution <!-- TODO: screenshot or hosted link -->
- `data_processing` → `model_training` → `model_evaluation`
- Done = `kedro run` gives AUC 0.6512 + your graph looks like this

**Say**
- This is what you'll build in the hands-on
- Point at the three pipelines and the datasets between them

---

## 6 — Live: Spaceflights

**Slide**
- `kedro new --starter spaceflights-pandas`
- `kedro run` · `kedro viz run`

**Say** (live, prepared and `uv sync`-ed beforehand)
- Log: load datasets → run nodes → save datasets, in an order nobody wrote
- Viz: click a node, a dataset, show the code panel

---

## 7 — Project structure & Data Catalog

**Slide**
- `conf/base/catalog.yml` — where data lives
- `conf/base/parameters.yml` — settings
- `conf/local/` — your machine, credentials, never committed
- `src/.../pipelines/<name>/nodes.py` + `pipeline.py`
- Catalog example:
  ```yaml
  churn_model:
    type: pickle.PickleDataset
    filepath: data/06_models/churn_classifier.pkl
    versioned: true
  ```

**Say**
- Same layout in every Kedro project → you always know where to look
- Every input/output declared once, with a name and a type; code never calls `read_csv` / `to_csv`
- Not declared = stays in memory during the run
- `versioned: true` → one timestamped copy per run → fixes "model overwritten"
- Replaces: personal paths

---

## 8 — Parameters

**Slide**
- `churn_window_days: 30` in `parameters.yml`
- Node input: `"params:churn_window_days"`
- Override: `kedro run --params "n_estimators=50"`

**Say**
- All magic numbers in one named place
- Experiment without editing code
- Replaces: `30` twice, `"Monthly"`, hardcoded hyperparameters

---

## 9 — Nodes & pipelines

**Slide**
- Node = function + named inputs + named outputs
  ```python
  Node(func=build_target,
       inputs=["members", "checkins", "params:churn_window_days"],
       outputs="members_with_target",
       name="build_target_node")
  ```
- Pipeline = list of nodes — **Kedro works out the order**
- Several return values → list of outputs

**Say**
- The function knows nothing about Kedro → unit-testable
- Output name = next node's input name, that's how they connect
- Replaces: cell order in the author's head

---

## 10 — Live: first node together

**Slide**
- Catalog → parameter → node → run → graph

**Say** (live, in the workshop repo)
1. `kedro run` → `Pipeline contains no nodes` — expected
2. `catalog.yml`: `members`, `checkins` already declared
3. Add `churn_window_days: 30` to `parameters.yml`
4. Uncomment the `build_target` node in `data_processing/pipeline.py`
5. `kedro run` → one node runs
6. `kedro viz run` → one node in the graph
- "That's the whole loop. Now you do the rest."

---

## 11 — Your turn (on screen for the whole exercise)

**Slide**
- 1. Messy notebook — fix the paths, answer the 📌 questions (~10 min)
- 2. Clean notebook — read it, same results (~5 min)
- 3. Kedro — catalog, parameters, 3 × `pipeline.py` (~30 min)
- Done = AUC 0.6512 in `data/08_reporting/metrics.json` + graph of slide 5
- Help: README → Quick Reference, Troubleshooting
- Solution shared at 1:15

**Say**
- Functions are already in `nodes.py` — you only write config and wiring
- Announce: **0:45** be in the Kedro project · **1:00** `data_processing` runs · **1:10** five minutes left

---

## 12 — Solution & before / after

**Slide**

| Notebook | Kedro |
|---|---|
| Personal paths | `catalog.yml` |
| Magic numbers | `parameters.yml`, `--params` |
| Cell order | graph from inputs/outputs |
| Charts on screen | `08_reporting/*.png` |
| Model overwritten | `versioned: true` |
| Run all and pray | `kedro run`, `--pipeline`, `--nodes` |

**Say**
- Share the Solutions repo, run `kedro run` + `kedro viz run`, walk the graph left to right
- `feature_cols` is one parameter used by training and the importance plot → no duplicated lists
- Only function change: `plot_roc_curve` takes the `metrics` dict — a node can only consume datasets

---

## 13 — Kedro at AXA Direct Assurance

**Slide**
- A dozen models on Kedro in Databricks
- Automated retraining inside our data ecosystem
- Raw data → retrained, versioned model in production
- <!-- TODO: simple architecture diagram -->
- Going further: namespaces, `kedro-mlflow`, deployment (Databricks, Airflow), `kedro jupyter lab`

**Say**
- What changed for the team: onboarding on someone else's model, code review, retraining
- <!-- TODO: one concrete number or anecdote -->
- README "Going Further": versioning, metrics history, namespaces

---

## 14 — Recap & thank you

**Slide**
- **Catalog** — where data lives
- **Parameters** — what you can change
- **Nodes** — what each step does
- **Pipelines** — how steps connect
- **Kedro-Viz** — see it all
- QR codes: workshop repo, Solutions repo, hosted Kedro-Viz <!-- TODO -->
- GitHub / LinkedIn <!-- TODO -->

**Say**
- One thing to remember: keep functions pure, let config describe the rest
- Questions

---

## Facilitator notes (not on slides)

**Milestones while circulating**
- `data_processing` (4 nodes) → Viz shows `members` + `checkins` → `model_input_table`
- `model_training` (3 nodes) → `churn_model` in Viz
- `model_evaluation` (4 nodes) → `metrics.json` = 0.6512…

**Where people get stuck**
- `params:… not found` → missing / misspelled in `parameters.yml`
- `X_test not found` with `--pipeline=model_evaluation` → inputs were in memory, run everything
- `returned by more than one nodes` → duplicate output name
- `Failed to save outputs` → outputs count ≠ return values
- `plot_roc_curve` needs `y_test`, `y_proba` **and** `metrics`
- `plot_feature_importance` returns two values
- Series as CSV → DataFrame → use `pickle.PickleDataset`
- `kedro` not recognised → `uv run kedro …`
- Notebook can't find data → kernel not on `.venv`, or path not `../data/01_raw/...`

**Likely questions**
- vs Airflow → Airflow orchestrates, Kedro structures code; `kedro-airflow` deploys to it
- vs MLflow → complementary, `kedro-mlflow`
- Databricks → yes, that's how AXA runs it
- Keep notebooks? → for exploration; `kedro jupyter lab` gives you `catalog`
- Testing → nodes are plain functions → `pytest`
- Credentials → `conf/local/credentials.yml`, git-ignored

**If running late**
- 1:00 → latecomers skip the messy notebook
- 1:15 → show the solution graph only
- Same error for many people → solve it once on screen

**Before the session**
- [ ] Repo URL final in README
- [ ] Spaceflights created and `uv sync`-ed
- [ ] Workshop repo in starter state for slide 10
- [ ] Viz screenshot / hosted link
- [ ] ZIP of both repos on a USB stick
- [ ] Large fonts, notifications off, one timed rehearsal
