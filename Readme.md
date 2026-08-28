# Freight Rate Prediction

Predicts freight `posted_rate` using load features (distance, equipment, weight, date, market signals).

## Approach
- EDA → Cleaning (missing values, outlier capping) → Feature Engineering → Time-based Train/Test Split → XGBoost Model → Predictions

## How to run
1. Install dependencies:
   pip install -r requirements.txt
2. Open `freight_rate_prediction.ipynb` in Jupyter or Google Colab
3. Upload the following data files when prompted: `data/train_test.csv`, `data/validation.csv`, `data/validation_predictions_template.csv`, `data/december_chart_inputs.csv`
4. Run all cells top to bottom
5. Outputs: `validation_predictions.csv` and `data/december_chart_inputs.csv` (filled)
6. Run scoring (provided separately by Spotter):
   python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv

## Results
- MAE: ~113 | RMSE: ~343 (on Sep-Oct held-out test set)
- Key predictor: distance (92.6% feature importance)
- Limitation: no Nov/Dec historical data — December predictions extrapolate, showing a weekly pattern rather than true seasonal trend
