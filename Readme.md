# Freight Rate Prediction
Predicts freight `posted_rate` using load features (distance, equipment, weight, date, market signals).

## Approach
- EDA → Cleaning (missing values, outlier capping) → Feature Engineering → Time-based Train/Test Split → XGBoost Model → Predictions

## Data
Place these files in a `data/` folder (not included in this repo):
- train_test.csv
- validation.csv
- validation_predictions_template.csv
- december_chart_inputs.csv

## How to run
1. Install dependencies:
   pip install -r requirements.txt
2. Open `freight_rate_prediction.ipynb` in Jupyter or Google Colab
3. Run all cells top to bottom
4. Outputs: `validation_predictions.csv` (root folder) and `data/december_chart_inputs.csv` (filled)
5. Run scoring:
   python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv

## Results
- MAE: ~113 | RMSE: ~343 (on Sep-Oct held-out test set)
- Key predictor: distance (92.6% feature importance)
- Limitation: no Nov/Dec historical data — December predictions extrapolate, showing a weekly pattern rather than true seasonal trend
