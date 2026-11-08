# Bulldozer Price Prediction

A notebook for Kaggle's Blue Book for Bulldozers competition: predict the auction sale price of used heavy equipment with a random forest.

## Data

The training table has 412,698 sales and 53 columns, mostly categorical machine configuration (drive system, enclosure, hydraulics, blade type) with a lot of missing values. `SalePrice` is the target.

The data files are the Kaggle competition files, kept in `train/`, `valid/` and `test/`, plus `Data Dictionary.xlsx` with the column definitions. The large CSVs are Git LFS objects. If you clone without Git LFS installed you get pointer files, not data, so run `git lfs install` before cloning (or `git lfs pull` afterwards).

`train/train_tmp.*`, `median_benchmark.csv`, `random_forest_benchmark_test.csv` and `test_predictions.csv` are not used by the notebook.

## What the notebook does

1. Loads `TrainAndValid.csv` with `saledate` parsed as a date and sorts by date.
2. Adds sale year, month, day, day of week and day of year, then drops `saledate`.
3. Fills missing numeric values with the column median and converts categorical columns to integer codes (missing becomes 0).
4. Trains a `RandomForestRegressor`, validating on a time split: everything before 2012 for training, 2012 sales for validation.
5. Scores with RMSLE, the competition metric, and R2.
6. Tries a smaller forest (`max_samples=10000`) and a `RandomizedSearchCV` run.
7. Applies the same preprocessing to `Test.csv` and builds a `SalesID` / `SalePrice` table.

## Results

From the notebook output, on the 2012 validation split:

| Model | Valid R2 | Train RMSLE | Valid RMSLE |
|---|---|---|---|
| Random Forest, default | 0.873 | 0.084 | 0.255 |
| Random Forest, `max_samples=10000` | 0.832 | 0.258 | 0.293 |
| Random Forest, `RandomizedSearchCV` best | 0.794 | 0.305 | 0.343 |

The search ran with `n_iter=2`, so it tried only two configurations and ended up behind the default model. The default forest is the one used for the test predictions.

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook "Bulldozer Price Prediction.ipynb"
```

The notebook reads `Bulldozer/TrainAndValid.csv` and `Bulldozer/Test.csv` (the path it had on Colab). Change those to `train/TrainAndValid.csv` and `test/Test.csv`, or put the files in a local `Bulldozer/` folder. A full-data forest fit takes a few minutes.
