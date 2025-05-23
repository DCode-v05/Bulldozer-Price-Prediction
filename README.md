# 🚜 Bulldozer Price Prediction

A comprehensive machine learning project to predict the sale price of used bulldozers based on auction data and machine attributes.

> 📊 Performed detailed EDA + 🧠 Trained & Tuned Random Forest Regression Models + 🔍 Handled Missing Data and Date Features

---

## 🗂️ Dataset Overview

- **Source**: [Bulldozer auction dataset (TrainAndValid.csv, Test.csv)](https://www.kaggle.com/competitions/bluebook-for-bulldozers)
- **Goal**: Predict the sale price (`SalePrice`) of bulldozers at auction
- **Target Column**: `SalePrice`

### 📋 Feature Descriptions

| Feature                   | Description |
|---------------------------|-------------|
| SalesID                   | Unique identifier of each auction sale |
| MachineID                 | Identifier for each machine (may have multiple sales) |
| ModelID                   | Unique machine model identifier |
| datasource                | Source of the sale record |
| auctioneerID              | Identifier of auctioneer company |
| YearMade                  | Manufacturing year of the machine |
| MachineHoursCurrentMeter  | Operating hours at sale time |
| UsageBand                 | Usage level compared to average for model (Low, Medium, High) |
| saledate                  | Date of sale |
| SalePrice                 | 💰 Target: Sale price in USD |
| fiModelDesc               | Description of machine model |
| ProductSize               | Size classification of product |
| ProductClassDesc          | Hierarchical product group description |
| State                     | US state where sale occurred |
| Drive_System              | Machine configuration details |
| Engine_Horsepower         | Engine horsepower rating |
| ...                       | Many other machine configuration features |

---

## 📊 Exploratory Data Analysis (EDA)

- Loaded data and checked structure with `.info()` and `.describe()`
- Visualized sale prices over time and distribution of sale prices
- Converted `saledate` from string to datetime format for temporal feature extraction
- Created new features:
  - ✅ Sale Year, Month, Day, Day of Week, Day of Year
  - ✅ Machine age and usage features (implicitly via `YearMade` and `MachineHoursCurrentMeter`)
- Checked and handled missing values for both numerical and categorical features
- Converted categorical variables to ordered categories and then numerical codes for modeling

---

## ⚙️ Machine Learning Models Used

## 🔍 Model Performance

| Model                         | Train RMSLE       | Valid RMSLE       |
|-------------------------------|-------------------|-------------------|
| Random Forest Regressor       | 0.0842            | 0.2547            |
| Tuned Random Forest Regressor | 0.3049            | 0.3427            |


> Used RMSLE (Root Mean Squared Log Error) to evaluate model performance on validation set.

---

## 🛠️ Hyperparameter Tuning

- Performed **RandomizedSearchCV** on Random Forest with parameters:
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [10, 20, 30]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
  - `max_features`: ['sqrt', 'log2']
  - `max_samples`: [10000]

- Resulted in better validation scores and model generalization.

---

## ▶️ How to Run

1. **Clone this repository**:
   ```bash
   git clone https://github.com/Denistanb/Bulldozer-Price-Prediction.git
   cd Bulldozer-Price-Prediction
2. **Install required libraries**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
3. **Run the Jupyter Notebook or Python script**:
   ```bash
     jupyter notebook "Bulldozer Price Prediction.ipynb"
