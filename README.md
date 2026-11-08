# Bulldozer Price Prediction

**Predicting the auction sale price of used heavy equipment from Kaggle's "Bluebook for Bulldozers" dataset, end to end in one notebook.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white)

## Overview

This is a regression project on the Kaggle "Bluebook for Bulldozers" competition. Given the configuration and auction history of a piece of heavy equipment, it predicts what the machine sold for. The data is a single combined `TrainAndValid` table of **412,698 sales x 53 columns**. Each row is one machine sold at one auction, and the target is `SalePrice` in USD.

Everything lives in one Jupyter notebook (`Bulldozer Price Prediction.ipynb`, 75 cells). It covers the usual steps: load, explore, engineer date features, fill missing values, encode categoricals, train a Random Forest, evaluate with the competition's RMSLE metric, tune hyperparameters, and score the test set. It's a learning project that follows a Zero-to-Mastery style template, with a time-based validation split, the competition's error metric and real reported numbers.

The catch with this dataset is that it's almost all categorical machine-configuration columns (drive system, enclosure, hydraulics, blade type and about 40 others), and most of them are heavily missing. A lot of the work is turning that into something a tree model can train on.

## Key Features

- Loads the combined Kaggle train/valid set (412,698 rows, 53 raw columns) with `low_memory=False` and date parsing on `saledate`.
- Exploratory analysis: dataframe `info()`/`describe()`, a per-column missing-value count, a sale-date vs. sale-price scatter, and a sale-price histogram.
- Date feature engineering: sorts records chronologically, derives **Sale Year, Sale Month, Sale Day, Sale Day of Week and Sale Day of Year** from `saledate`, and drops the original column.
- Missing values handled separately for the two column types: numeric columns are filled with the column **median**, and categorical columns become integer codes (with `+1` so missing becomes its own `0` category).
- Categorical encoding with pandas `Categorical.codes`, which keeps all the machine-configuration columns usable without blowing up the number of features.
- **RandomForestRegressor** as the model, with a time-based train/validation split (train on everything before 2012, validate on 2012 sales) instead of a random split, since the test set is strictly in the future.
- **RMSLE (Root Mean Squared Log Error)** evaluation, the competition metric, next to R2 scoring.
- Hyperparameter tuning with `RandomizedSearchCV` (5-fold CV) over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` and `max_samples`.
- A `max_samples=10000` "reduced" model, so you can iterate in seconds instead of minutes.
- Applies the same preprocessing to the 12,457-row test set and builds a predictions table in the competition's `Sales Id` / `Sales Price` layout. The notebook does not write it to a file.

## How It Works

### Data

The repo holds the Kaggle competition files. The CSVs (`Train.csv`, `TrainAndValid.csv`, `Test.csv`, `Valid.csv`, `ValidSolution.csv` and the others) are **Git LFS pointer files** in this checkout, while the `.xlsx` workbooks under `train/` (about 40 MB each) are committed in full. There is also a `Data Dictionary.xlsx` and `Machine_Appendix.csv` for reference, plus Kaggle's `median_benchmark.csv` and `random_forest_benchmark_test.csv` baselines. The repo has no `.gitattributes`, so LFS has to be set up on your side to get the real CSV contents.

The notebook reads `TrainAndValid.csv`. `SalePrice` runs from $4,750 to $142,000 with a mean around $31,215. Missingness is severe in places: `MachineHoursCurrentMeter` is null in 265K of 412K rows, `UsageBand` in 339K, and a dozen blade/engine columns are missing in about 387K rows.

### Preprocessing

1. **Parse and sort dates.** `saledate` is read as a real datetime and the frame is sorted ascending, so the split later is chronological.
2. **Engineer date parts.** Year / month / day / day-of-week / day-of-year are pulled off `saledate`, then `saledate` itself is dropped. This gives the trees a time trend to learn from.
3. **Object to category.** Every `object` column is cast to an ordered pandas `category`.
4. **Fill numeric gaps.** For each numeric column with nulls, fill with that column's **median** (for example `auctioneerID`, `MachineHoursCurrentMeter`).
5. **Encode categoricals.** Each non-numeric column becomes `pd.Categorical(value).codes + 1`. The `+1` moves pandas' `-1` "missing" marker to `0`, so a missing category is just another integer the forest can split on. After this step there are no nulls and 57 numeric columns.

### Model and validation

The target/features split is `X = dozer.drop("SalePrice")`, `y = dozer["SalePrice"]`. Training uses `RandomForestRegressor(n_jobs=-1)` with a fixed `np.random.seed(30)`.

A first fit on the *entire* dataset scores R2 0.988, but it is scored on data it trained on, so it only works as a sanity baseline. The real evaluation uses a time-based split: `train = rows where Sale Year != 2012`, `valid = rows where Sale Year == 2012`. That follows the competition setup (train through 2011, validate on 2012) and keeps future sales out of training.

### Evaluation

The scoring helper reports Train and Valid **RMSLE** using scikit-learn's `root_mean_squared_log_error`. RMSLE fits here because it penalizes relative error rather than absolute dollars: being off by $5K on a $100K machine shouldn't count the same as being off by $5K on a $7K machine.

### Tuning

`RandomizedSearchCV` (5-fold) searches the hyperparameter grid with `max_samples=10000` to keep each fit fast. The search runs with `n_iter=2`, so it only tries two candidate configurations. That is enough to show the workflow but not a real sweep, which is why the tuned model ends up slightly *behind* the default. The final `ideal_model` is the plain default Random Forest, the best of the validated models.

### Prediction

The 12,457-row `Test.csv` goes through the same date-feature, median-fill and categorical-code steps, the column counts are checked to match (56 each), and `ideal_model.predict` produces the sale-price estimates. They are put into a `Sales Id` / `Sales Price` dataframe in the Kaggle submission layout.

## Results / Highlights

| Model | Valid R2 | Train RMSLE | Valid RMSLE |
|---|---|---|---|
| Random Forest (full data, no split) | 0.988 | — | — |
| **Random Forest (time-based split)** | **0.873** | **0.0842** | **0.2547** |
| Random Forest (`max_samples=10000`) | 0.832 | 0.2576 | 0.2931 |
| Random Forest (RandomizedSearchCV best) | 0.794 | 0.3049 | 0.3427 |

- The default Random Forest reaches **R2 0.873** on 2012 sales with a **validation RMSLE of 0.255**.
- Best tuned params found: `{n_estimators: 300, max_depth: 20, max_features: 'sqrt', min_samples_split: 5, min_samples_leaf: 1, max_samples: 10000}`.
- Cutting to a 10K-sample forest takes training from about 3.5 min wall time to about 8 seconds, and valid R2 only drops from 0.873 to 0.832. That's a good trade while iterating.
- The gap between Train RMSLE (0.084) and Valid RMSLE (0.255) is the overfitting you'd expect from an untuned forest on 412K rows.

## Tech Stack

- **Language:** Python 3
- **Notebook:** Jupyter
- **Data:** pandas, NumPy
- **ML:** scikit-learn (`RandomForestRegressor`, `RandomizedSearchCV`, `root_mean_squared_log_error`)
- **Visualization:** Matplotlib, Seaborn
- **Storage:** Git LFS (for the large CSV data files)

## Getting Started

### Prerequisites
- Python 3.x and Jupyter
- Git LFS installed (the CSV data files are LFS-tracked; without it you'll clone the pointer files, not the data)

### Installation
```bash
git lfs install
git clone https://github.com/DCode-v05/Bulldozer-Price-Prediction.git
cd Bulldozer-Price-Prediction
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

If you cloned before installing Git LFS, run `git lfs pull` to fetch the CSV contents.

### Running
```bash
jupyter notebook "Bulldozer Price Prediction.ipynb"
```

## Usage

Open the notebook and run the cells top to bottom. The flow is: load `TrainAndValid` -> explore -> engineer date features -> fill missing values -> encode categoricals -> train the Random Forest -> evaluate with RMSLE -> tune -> predict on the test set.

A few things to know:

- The notebook reads its CSVs from a `Bulldozer/` path (for example `pd.read_csv("Bulldozer/TrainAndValid.csv")`), the path it had on Colab. The committed files live in `train/`, `test/` and `valid/`, so change the read paths to `train/TrainAndValid.csv` and `test/Test.csv`, or drop the CSVs into a local `Bulldozer/` folder.
- Full-data training takes a few minutes per fit. If you're just iterating, use the `max_samples=10000` variant. It's nearly as accurate and runs in seconds.
- The last cell builds a predictions dataframe in `Sales Id` / `Sales Price` format. The notebook doesn't save it, so write it out yourself if you want a Kaggle submission.
- `train/train_tmp.*`, `median_benchmark.csv`, `random_forest_benchmark_test.csv` and `test_predictions.csv` are not used by the notebook.

## Project Structure

```
Bulldozer-Price-Prediction/
├── Bulldozer Price Prediction.ipynb   # The whole pipeline: EDA, features, model, tuning, prediction (75 cells)
├── Data Dictionary.xlsx               # Kaggle's column definitions
├── Machine_Appendix.csv               # Supplementary machine reference data (LFS)
├── median_benchmark.csv               # Kaggle median benchmark (LFS)
├── random_forest_benchmark_test.csv   # Kaggle RF benchmark (LFS)
├── test_predictions.csv               # Saved model predictions (LFS)
├── train/
│   ├── Train.csv / Train.xlsx                  # Training data through 2011
│   ├── TrainAndValid.csv / TrainAndValid.xlsx  # Combined set used by the notebook
│   └── train_tmp.csv / train_tmp.xlsx          # Working copy
├── test/
│   └── Test.csv                       # 12,457 future sales to predict (LFS)
└── valid/
    ├── Valid.csv                      # Jan-Apr 2012 validation set
    └── ValidSolution.csv              # Validation ground truth
```

---

## Contact

<table>
  <tr><td><b>Portfolio:</b> <a href="https://www.denistan.me">Denistan</a></td><td><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/denistanb">denistanb</a></td></tr>
  <tr><td><b>GitHub:</b> <a href="https://github.com/DCode-v05">DCode-v05</a></td><td><b>LeetCode:</b> <a href="https://leetcode.com/u/Denistan_B">Denistan_B</a></td></tr>
  <tr><td colspan="2" align="center"><b>Email:</b> <a href="mailto:denistanb05@gmail.com">denistanb05@gmail.com</a></td></tr>
</table>

Made with ❤️ by **Denistan B**
