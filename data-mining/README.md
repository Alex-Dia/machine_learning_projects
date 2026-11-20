# Bank Marketing — Data Exploration & Custom KNN

**Context:** Part B Data Visualisation was done by team member. I have done the other parts as well as QA before finalising.

## What this project demonstrates

- Exploratory data analysis: distributions, attribute types, missing/unknown values, statistical outliers, duplicate detection.
- Classification workflow: duplicate removal, meaningful handling of `pdays=999`, removal of `duration` to avoid prediction-time leakage, a stratified train/test split, scaling and one-hot encoding.
- A custom Euclidean K-nearest-neighbours classifier and a comparison to scikit-learn KNN.
- K selection through cross-validation, classification metrics and a comparison with manually recorded WEKA IBk results.

## Dataset

**Bank Marketing**, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/222/bank+marketing

Obtain the **bank-additional-full.csv** variant (typically found inside the downloadable `bank-additional.zip` archive). This source dataset is not bundled in this repository. Place the CSV inside `notebooks/` because both original notebooks load `bank-additional-full.csv` from the working directory.

## Setup (macOS / Linux)

```bash
cd data-mining
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd notebooks
jupyter lab
```

Open `bank-marketing-eda.ipynb` first, then `custom-knn-bank-marketing.ipynb`. Run each notebook top-to-bottom with the working directory set to `notebooks/`.

## Reproducibility and review notes

1. **Important:** The custom KNN notebook computes `BEST_K` via cross-validation, but the final evaluation cells use `k=7` (custom) and `n_neighbors=7` (scikit-learn). Replace both hard-coded values with `BEST_K` *and rerun all dependent cells* if the intent is to report the tuned model. Do not present existing saved output as a verified tuned result before doing so.
2. The opening heading refers to a proposal and mentions a decision tree, but these included notebooks focus on exploratory analysis and KNN; update the heading to match the work actually shown.
3. The WEKA IBk comparison is entered manually in the code. It was not reproduced by this packaging step; document WEKA settings/source before claiming a fully reproducible three-way benchmark.
4. StandardScaler and OneHotEncoder use the training data in the original split; cross-validation of pre-transformed features deserves additional review for possible preprocessing leakage. For a rigorous model-selection comparison, wrap preprocessing and the estimator in a Pipeline and cross-validate that Pipeline.
5. Running the KNN implementation on the full dataset may take considerable time and memory.
