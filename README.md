# Income Classification & Customer Segmentation

Take-home project on the 1994–1995 Census-Income (KDD) dataset. Two deliverables: a classifier predicting whether a person earns above or below \$50K, and a customer segmentation model for marketing.

## Project structure

```
project/
├── data/
│   ├── censusbureau.data           # raw input — drop here before running
│   └── census-bureau.columns       # raw input — column names, one per line
├── notebooks/
│   ├── 0_eda.ipynb                 # data exploration
│   ├── 1_preprocessing.ipynb       # cleaning, feature engineering, encoding
│   ├── 2_classifier.ipynb          # classification + tuning + evaluation
│   ├── 3_segmentation.ipynb        # K-means segmentation + personas
│   └── 4_fairness.ipynb            # adverse impact analysis
├── reports/                        # generated plots and tables (created on first run)
├── decisions.md                    # log of non-obvious methodology choices
├── project_report.md               # main report deliverable
├── requirements.txt
└── README.md
```

The `data/` and `reports/` directories should exist next to `notebooks/`. The notebooks reference them as `../data` and `../reports` so they must be run from inside `notebooks/`.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

## Input data

Place the two raw files in `data/`:

- `censusbureau.data` — comma-delimited data file, no header row
- `census-bureau.columns` — column names, one per line, in the same order as the columns in the data file

These are the UCI Census-Income (KDD) files. Original source: https://archive.ics.uci.edu/dataset/117/census+income+kdd

No data is included in this repo. The notebooks will fail at the loading step if either file is missing.

## How to run

The five notebooks must be run in order. Each one saves intermediate artifacts to `data/` and final outputs (plots, tables) to `reports/`. Run them from inside `notebooks/` so the relative paths resolve correctly.

Interactive (recommended for reviewing):

```bash
cd notebooks
jupyter notebook
# then open and run each notebook top-to-bottom, in numerical order
```

Headless (for reproducing all outputs):

```bash
cd notebooks
for nb in 0_eda 1_preprocessing 2_classifier 3_segmentation 4_fairness; do
    jupyter nbconvert --to notebook --execute "${nb}.ipynb" --inplace
done
```

Total runtime is roughly 30 minutes on a modern laptop. The classifier notebook is the slowest (~15 min, mostly hyperparameter tuning and SHAP).

## Notebook dependencies

```
0_eda → 1_preprocessing → 2_classifier → 4_fairness
                       ↘ 3_segmentation
```

- `0_eda` is the foundation. Produces `df_loaded.parquet` which everything else loads.
- `1_preprocessing` produces the train/test splits and the fitted preprocessor.
- `2_classifier` produces the trained models and the scored test set.
- `3_segmentation` only needs `df_loaded.parquet`; can run any time after `0_eda`.
- `4_fairness` needs both `df_loaded.parquet` and the scored test set from `2_classifier`.

If you re-run any notebook, downstream notebooks should be re-run as well to keep artifacts consistent.

## Expected outputs

After a full run, `reports/` should contain:

EDA:
- `schema_summary.csv` — per-column dtype, cardinality, missing/NIU rates
- `mi_ranking.csv` — mutual-information ranking against the target

Classification:
- `pr_roc_curves.png` — PR and ROC curves for the three models
- `calibration.png` — calibration plot
- `gain_chart.png` — cumulative gain curve (headline marketing visual)
- `shap_summary.png` — SHAP feature importance
- `lift_table.csv` — decile-level lift, captured rate, and targeting efficiency

Segmentation:
- `k_selection.png` — elbow and silhouette plots
- `segment_pca.png` — 2D PCA projection of the clusters
- `segment_profile.csv` — per-segment feature means and population shares
- `segment_marketing_table.csv` — high-income concentration and targeting efficiency by segment
- `segment_personas.csv` — final segment-to-persona mapping

Fairness:
- `fairness_selection_rate.png` — selection rate by sex, race, Hispanic origin, citizenship

`data/` will also contain intermediate parquet files and serialized models (`*.joblib`) — these are loaded by downstream notebooks but aren't deliverables themselves.

## Reading order (for review without execution)

1. `project_report.md` — the main deliverable, written for the client
2. `decisions.md` — log of non-obvious methodology choices and tradeoffs
3. The notebooks themselves, in order, for the underlying work

## Notes

- Random seed is fixed at 42 across all notebooks for reproducibility.
- The survey instance weight is used as `sample_weight` in model training and evaluation, never as a feature. See `decisions.md` for the methodology.
- The fairness analysis (Section 7 of the report) flags substantial disparate-impact issues at the F1-optimal threshold. The model is not production-ready as-is; pre-deployment mitigation is required.
- LightGBM was selected as the final model based on population-weighted PR-AUC. The classifier and segmentation are designed to be used together: classifier scores prospects, segmentation tells the client which campaign to run for each prospect.
