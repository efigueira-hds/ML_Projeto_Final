# Predicting Post-Operative Knee-Alignment (CPAK) Phenotype Change

A machine-learning project that predicts — from **pre-operative data only** — whether a
knee-replacement patient's **CPAK** (Coronal Plane Alignment of the Knee) phenotype changes
after surgery. It is a **binary classification** problem on a **small, imbalanced clinical
dataset**, and the focus of the project is *methodological rigour*: leakage-free
preprocessing, honest cross-validation, probability-based evaluation, and principled handling
of class imbalance — all wrapped in reusable scikit-learn pipelines.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikitlearn&logoColor=white)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-SMOTE-2E7D32)
![pandas](https://img.shields.io/badge/pandas-data-150458?logo=pandas&logoColor=white)

> The notebooks and their inline documentation are written in **European Portuguese**
> (academic context). This README is in English for a broader audience.

---

## Why this project is worth a look

This is not just "fit three models and print accuracy". The pipeline is engineered to avoid
the mistakes that quietly inflate ML results:

- **No data leakage.** Preprocessing is split into two tiers: *deterministic / domain*
  transforms applied before the train/test split, and *learned* transforms (median
  imputation, scaling, SMOTE) that live **inside a `Pipeline`** and are therefore refit on
  the training fold only. The test set never influences imputation, scaling, or resampling.
- **Cross-validation done right.** Every `cross_val_predict` / `GridSearchCV` runs on the
  **training set only**; the held-out test set is touched exactly once, for the final report.
- **Probability-based metrics.** ROC-AUC is computed from `predict_proba`, not from hard
  labels — the correct way to score a ranking/threshold-independent metric.
- **Class imbalance handled explicitly.** The positive class is ~10% of the data. The project
  compares class weighting and **SMOTE** (via `imblearn.pipeline.Pipeline`, so synthetic
  samples are generated only during training, never in validation/test).
- **Model selection, not guessing.** Decision-tree depth and the other hyper-parameters are
  tuned by cross-validation, with a stated rationale — not hard-coded.
- **DRY & reproducible.** All shared logic lives in [`preprocessing.py`](preprocessing.py);
  each model notebook differs only in the estimator and its tuning. A fixed
  `random_state = 42` is used throughout.

---

## The problem

CPAK is a validated classification of knee phenotypes based on coronal-plane alignment. A
patient is assigned a CPAK group **before** surgery (`Grupo_pre`) and **after** surgery
(`Grupo_pos`). The prediction target is:

> **`mudanca_CPAK`** — did the patient's CPAK group change from pre- to post-operative?
> (`1` = changed, `0` = unchanged)

The task is to anticipate this change using **only information available before surgery**, so
all post-operative measurements are deliberately excluded from the feature set.

---

## Dataset

| | |
|---|---|
| Records | 265 raw → **262** after cleaning |
| Features used | **13** pre-operative variables |
| Target | `mudanca_CPAK` (binary) |
| Class balance | **~90 % / ~10 %** (236 unchanged / 26 changed) |

**Features:** demographics (age, sex, weight, height, BMI), the pre-operative CPAK group,
and baseline clinical measures — knee flexion range of motion, pain (VAS), 6-minute walk
test, and WOMAC-type functional sub-scores. Variables suffixed `_90` (post-operative) and
`Grupo_pos` are **excluded as leakage**.

**Data cleaning** (in [`preprocessing.py`](preprocessing.py)): `999` sentinel → `NaN`,
recoding of `Sexo` to `0/1`, recomputing `IMC` (BMI) from weight/height, reconstructing a
WOMAC total from its components, and dropping records without a valid target. Missing values
are **not** imputed here — that is left to the pipeline (see leakage note above).

> ⚠️ The raw clinical dataset (`ML_Ortopedia_CPAK.xlsx`) is **not committed** to the
> repository for data-privacy reasons.

---

## Repository structure

| Path | What it contains |
|---|---|
| [`preprocessing.py`](preprocessing.py) | Shared module: deterministic data loading, the `Pipeline`/`ColumnTransformer` builder, SMOTE integration, and evaluation helpers. |
| `1.EDA.ipynb` | Exploratory data analysis — distributions, outliers, missing-value strategy, correlation analysis, and statistical tests (Mann-Whitney U, Spearman) of each feature vs. the target. |
| `2.Modelação.ipynb` | Baseline comparison of the three models on a common pipeline. |
| `RL_limpo.ipynb` | **Logistic Regression** — the main model: tuning (`GridSearchCV`), coefficient interpretation, threshold analysis, SMOTE, and the final model. |
| `AD_final.ipynb` | **Decision Tree** — depth tuned by cross-validation to control overfitting. |
| `RF.ipynb` | **Random Forest** — feature-reduction study, tuning, and SMOTE. |

---

## Methodology

**Two-tier preprocessing** — deterministic transforms are safe before the split; learned
transforms are confined to the pipeline:

```
raw Excel ──(deterministic cleaning)──> X (with NaNs) ──train/test split──┐
                                                                          │
        ┌─── Pipeline (fit on train folds only) ───────────────────────────┘
        │    ColumnTransformer: median imputation (num) + one-hot (Grupo_pre)
        │      → StandardScaler (for Logistic Regression)
        │      → SMOTE (optional, imblearn — training folds only)
        │      → estimator
        └──────────────────────────────────────────────────────────────────
```

**Modelling:** a single stratified train/test split, baseline comparison of Logistic
Regression / Decision Tree / Random Forest, then per-model tuning via `GridSearchCV`
(stratified 5-fold CV on the training set, scored on ROC-AUC). The Logistic Regression is
the primary model, chosen for its interpretability and competitive performance; a decision
threshold is tuned on the training set to favour recall of the rare "changed" class.

---

## Results

Final performance, **ROC-AUC from predicted probabilities**. Given the small test set
(79 observations, 8 positives), the **5-fold cross-validated AUC on the training set** is the
more stable estimate of generalisation.

| Model | Test AUC | CV AUC (train) |
|---|:---:|:---:|
| Logistic Regression — baseline | 0.72 | 0.82 |
| Logistic Regression — **tuned (main model)** | 0.71 | **0.84** |
| Random Forest — baseline | 0.74 | 0.78 |
| Random Forest — reduced features + tuned | 0.68 | 0.81 |
| Decision Tree — baseline (unbounded, overfits) | 0.56 | 0.59 |
| Decision Tree — depth tuned (`max_depth = 4`) | 0.54 | 0.78 |

**Reading the results.** Limiting the decision tree's depth lifts CV-AUC from **0.59 → 0.78**
while eliminating the tell-tale train-accuracy-of-1.0 overfit — a clear demonstration of
regularisation. SMOTE improved test AUC in some configurations but was **unstable across
folds** given the tiny minority class, and is reported transparently rather than cherry-picked.
The confusion-matrix analysis in each notebook shows the precision/recall trade-off for the
rare class, which matters more here than headline accuracy on an imbalanced target.

---

## Reproducing the analysis

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn openpyxl jupyter

# 2. Place the dataset (not included) in the project root
#    ML_Ortopedia_CPAK.xlsx

# 3. Run the notebooks in order (each runs top-to-bottom from a clean kernel):
#    1.EDA.ipynb  →  2.Modelação.ipynb  →  RL_limpo.ipynb / AD_final.ipynb / RF.ipynb
jupyter notebook
```

The model notebooks import `preprocessing.py` and load the raw Excel directly, so they do
**not** depend on any intermediate CSV.

---

## Tech stack

**Python** · **pandas** / **NumPy** (data) · **scikit-learn** (pipelines, models, model
selection, metrics) · **imbalanced-learn** (SMOTE) · **matplotlib** / **seaborn**
(visualisation) · **SciPy** (statistical tests) · **Jupyter**.

---

## Author

**Eduardo** · GitHub: [@Edweirdo-pt](https://github.com/Edweirdo-pt)
<!-- Add your LinkedIn / email here if you want recruiters to reach you directly. -->
