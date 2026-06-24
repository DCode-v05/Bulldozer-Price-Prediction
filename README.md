# Bulldozer Price Prediction

**Predicting the auction sale price of used heavy equipment from Kaggle's "Bluebook for Bulldozers" dataset, end to end in one notebook.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white) ![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=matplotlib&logoColor=white)

## Overview

This is a regression project on the Kaggle "Bluebook for Bulldozers" competition: given the configuration and auction history of a piece of heavy equipment, predict what it sold for. The data is a single combined `TrainAndValid` table of **412,698 sales x 53 columns**, where each row is one machine sold at one auction and the target is `SalePrice` in USD.

The whole thing lives in one Jupyter notebook (`Bulldozer Price Prediction.ipynb`, 75 cells) and walks through the standard pipeline: load, explore, engineer date features, fill missing values, encode categoricals, train a Random Forest, evaluate with the competition's RMSLE metric, tune hyperparameters, and finally score the held-out test set. It's a learning project that follows a Zero-to-Mastery style template, but the methodology holds up end to end — proper time-based validation split, the right error metric, and real reported numbers.

The catch with this dataset is that it's almost entirely categorical machine-configuration columns (drive system, enclosure, hydraulics, blade type, and ~40 others), most of them heavily missing. A lot of the work is turning that into something a tree model can actually train on.

## Key Features

- Loads the combined Kaggle train/valid set (412,698 rows, 53 raw columns) with `low_memory=False` and date parsing on `saledate`.
- Exploratory analysis: dataframe `info()`/`describe()`, a per-column missing-value count, a sale-date vs. sale-price scatter, and a sale-price histogram.
- Date feature engineering: sorts records chronologically, then derives **Sale Year, Sale Month, Sale Day, Sale Day of Week, and Sale Day of Year** from `saledate`, and drops the original column.
- Missing-value handling done separately for the two column types — numeric columns filled with the column **median**, categorical columns turned into integer codes (with `+1` so missing becomes its own `0` category).
- Categorical encoding via pandas `Categorical.codes`, which keeps all ~40 machine-configuration columns usable without exploding dimensionality.
- **RandomForestRegressor** as the model, with a proper time-based train/validation split (train on everything before 2012, validate on 2012 sales) instead of a random split — which matters because the test set is strictly in the future.
- **RMSLE (Root Mean Squared Log Error)** evaluation function — the actual competition metric — wrapped alongside R2 scoring.
- Hyperparameter tuning with `RandomizedSearchCV` (5-fold CV) over `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, and `max_samples`.
- A `max_samples=10000` "reduced" model so you can iterate on training in seconds instead of minutes.
- Applies the identical preprocessing to the 12,457-row test set and writes out predictions in the competition's `SalesID, SalePrice` format.

## How It Works

### Data

The repo ships the Kaggle competition files directly. The large CSVs (`Train.csv`, `TrainAndValid.csv`, `Test.csv`) are stored as **Git LFS pointers** — the real `Train.csv` is ~116 MB — while the equivalent `.xlsx` workbooks (~40 MB each) are committed in full under `train/`. There's also a `Data Dictionary.xlsx` and `Machine_Appendix.csv` for reference, plus Kaggle's `median_benchmark.csv` and `random_forest_benchmark_test.csv` baselines.

The notebook reads `TrainAndValid.csv`. Quick shape of the target: `SalePrice` runs from $4,750 to $142,000, mean around $31,215. Missingness is severe in places — `MachineHoursCurrentMeter` is null in 265K of 412K rows, `UsageBand` in 339K, and a dozen blade/engine columns are missing in ~387K rows.

### Preprocessing

1. **Parse and sort dates.** `saledate` is read as a real datetime and the frame is sorted ascending, so the split later is genuinely chronological.
2. **Engineer date parts.** Year / month / day / day-of-week / day-of-year are pulled off `saledate`, then `saledate` itself is dropped. This is what gives the tree something to learn the time trend from.
3. **Object to category.** Every `object` column is cast to an ordered pandas `category`.
4. **Fill numeric gaps.** For each numeric column with nulls, fill with that column's **median** (e.g. `auctioneerID`, `MachineHoursCurrentMeter`).
5. **Encode categoricals.** Each non-numeric column becomes `pd.Categorical(value).codes + 1`. The `+1` shifts pandas' `-1` "missing" sentinel up to `0`, so a missing category is just another integer the forest can branch on. After this step there are zero nulls and 57 numeric columns.

### Model and validation

The target/features split is `X = dozer.drop("SalePrice")`, `y = dozer["SalePrice"]`. Training uses `RandomForestRegressor(n_jobs=-1)` with a fixed `np.random.seed(30)`.

A first fit on the *entire* dataset scores R2 0.988 — but that's training on data it's also scoring, so it's only a sanity baseline. The real evaluation uses a time-based split: `train = rows where Sale Year != 2012`, `valid = rows where Sale Year == 2012`. That mirrors the competition setup (train through 2011, validate on 2012) and avoids leaking future sales into training.

### Evaluation

The scoring helper reports both Train and Valid **RMSLE** using scikit-learn's `root_mean_squared_log_error`. RMSLE is the right metric here because it penalizes relative error rather than absolute dollars — being off by $5K on a $100K machine shouldn't count the same as being off by $5K on a $7K machine.

### Tuning

`RandomizedSearchCV` (5-fold) searches the hyperparameter grid with `max_samples=10000` to keep each fit fast. Note: the search is run with `n_iter=2`, so it only tries two candidate configurations — enough to demonstrate the workflow, but not a thorough sweep, which is why the tuned model ends up slightly *behind* the default. The final `ideal_model` is therefore set to the plain default Random Forest, which is the strongest validated model.

### Prediction

The 12,457-row `Test.csv` is run through the exact same date-feature + median-fill + categorical-code steps, the column counts are checked to match (56 each), and `ideal_model.predict` produces sale-price estimates. Those are assembled into a `Sales Id / Sales Price` dataframe matching the Kaggle submission format.

## Results / Highlights

| Model | Valid R2 | Train RMSLE | Valid RMSLE |
|---|---|---|---|
| Random Forest (full data, no split) | 0.988 | — | — |
| **Random Forest (time-based split)** | **0.873** | **0.0842** | **0.2547** |
| Random Forest (`max_samples=10000`) | 0.832 | 0.2576 | 0.2931 |
| Random Forest (RandomizedSearchCV best) | 0.794 | 0.3049 | 0.3427 |

- The validated default Random Forest reaches **R2 0.873** on 2012 sales with a **validation RMSLE of 0.255** — a solid score for this competition.
- Best tuned params found: `{n_estimators: 300, max_depth: 20, max_features: 'sqrt', min_samples_split: 5, min_samples_leaf: 1, max_samples: 10000}`.
- Reducing to a 10K-sample forest cuts training from ~3.5 min wall time down to ~8 seconds while only dropping valid R2 from 0.873 to 0.832 — a useful speed/accuracy trade for iteration.
- Gap between Train RMSLE (0.084) and Valid RMSLE (0.255) shows the expected amount of overfitting for an untuned forest on 412K rows.

## Tech Stack

- **Language:** Python 3
- **Notebook:** Jupyter
- **Data:** pandas, NumPy
- **ML:** scikit-learn — `RandomForestRegressor`, `RandomizedSearchCV`, `root_mean_squared_log_error`
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

### Running
```bash
jupyter notebook "Bulldozer Price Prediction.ipynb"
```

## Usage

Open the notebook and run the cells top to bottom. The flow is: load `TrainAndValid` -> explore -> engineer date features -> fill missing values -> encode categoricals -> train the Random Forest -> evaluate with RMSLE -> tune -> predict on the test set.

A couple of things worth knowing:

- The notebook reads its CSVs from a `Bulldozer/` path (e.g. `pd.read_csv("Bulldozer/TrainAndValid.csv")`). The committed files live in `train/`, `test/`, and `valid/`, so point the read paths at wherever you've placed the data (or drop the CSVs into a local `Bulldozer/` folder).
- Full-data training takes a few minutes per fit. If you're just iterating, use the `max_samples=10000` variant — it's nearly as accurate and runs in seconds.
- The final cell produces a predictions dataframe in `Sales Id / Sales Price` format that you can write out for a Kaggle submission.

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

**Portfolio:** [Denistan](https://www.denistan.me)<br>
**LinkedIn:** [Denistan](https://www.linkedin.com/in/denistanb)<br>
**GitHub:** [DCode-v05](https://github.com/DCode-v05)<br>
**LeetCode:** [Denistan_B](https://leetcode.com/u/Denistan_B)<br>
**Email:** [denistanb05@gmail.com](mailto:denistanb05@gmail.com)

Made with ❤️ by **Denistan B**
