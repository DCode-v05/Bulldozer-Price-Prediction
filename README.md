# Bulldozer Price Prediction

## Project Description
A comprehensive machine learning project aimed at predicting the sale price of used bulldozers based on auction data and machine attributes. This project leverages data science and machine learning techniques to provide accurate price predictions, assisting stakeholders in making informed decisions during equipment auctions.

---

## Project Details

### Dataset Overview
- **Source:** [Kaggle - Bluebook for Bulldozers](https://www.kaggle.com/competitions/bluebook-for-bulldozers)
- **Goal:** Predict the sale price (`SalePrice`) of bulldozers at auction.
- **Target Column:** `SalePrice`

#### Feature Descriptions (Partial)
- **SalesID:** Unique identifier of each auction sale
- **MachineID:** Identifier for each machine (may have multiple sales)
- **ModelID:** Unique machine model identifier
- **datasource:** Source of the sale record
- **auctioneerID:** Identifier of auctioneer company
- **YearMade:** Manufacturing year of the machine
- **MachineHoursCurrentMeter:** Operating hours at sale time
- **UsageBand:** Usage level compared to average for model (Low, Medium, High)
- **saledate:** Date of sale
- **SalePrice:** Target: Sale price in USD
- **fiModelDesc:** Description of machine model
- **ProductSize:** Size classification of product
- **ProductClassDesc:** Hierarchical product group description
- **State:** US state where sale occurred
- **Drive_System:** Machine configuration details
- **Engine_Horsepower:** Engine horsepower rating
- ... (many other machine configuration features)

### Exploratory Data Analysis (EDA)
- Data loading and structure inspection
- Visualization of sale prices and their distribution
- Feature engineering (date features, machine age, usage)
- Handling missing values
- Categorical variable encoding

### Machine Learning Approach
- **Model Used:** Random Forest Regressor
- **Evaluation Metric:** RMSLE (Root Mean Squared Log Error)
- **Hyperparameter Tuning:** RandomizedSearchCV with various parameters

#### Model Performance
| Model                         | Train RMSLE | Valid RMSLE |
|-------------------------------|-------------|-------------|
| Random Forest Regressor       | 0.0842      | 0.2547      |
| Tuned Random Forest Regressor | 0.3049      | 0.3427      |

---

## Tech Stack
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

---

## Getting Started

1. **Clone this repository:**
   ```bash
   git clone https://github.com/TensoRag/Bulldozer-Price-Prediction.git
   cd Bulldozer-Price-Prediction
   ```
2. **Install required libraries:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. **Run the Jupyter Notebook:**
   ```bash
   jupyter notebook "Bulldozer Price Prediction.ipynb"
   ```

---

## Usage
- Open the Jupyter Notebook and follow the step-by-step analysis and modeling process.
- Modify the code to experiment with different models or feature engineering techniques.
- Use the provided datasets in the `train/`, `test/`, and `valid/` folders for your own experiments.

---

## Project Structure
```
Bulldozer-Price-Prediction/
├── Bulldozer Price Prediction.ipynb      # Main analysis and modeling notebook
├── Data Dictionary.xlsx                  # Data dictionary for features
├── Machine_Appendix.csv                  # Additional machine data
├── median_benchmark.csv                  # Benchmark results
├── random_forest_benchmark_test.csv      # Model test results
├── test_predictions.csv                  # Predictions on test set
├── train/
│   ├── Train.csv
│   ├── Train.xlsx
│   ├── TrainAndValid.csv
│   ├── TrainAndValid.xlsx
│   ├── train_tmp.csv
│   └── train_tmp.xlsx
├── test/
│   └── Test.csv
├── valid/
│   ├── Valid.csv
│   └── ValidSolution.csv
```

---

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to your branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request describing your changes.

---

## Contact
- **GitHub:** [TensoRag](https://github.com/TensoRag)
- **Email:** denistanb05@gmail.com
