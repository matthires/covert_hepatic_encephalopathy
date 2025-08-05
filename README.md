# Covert Hepatic Encephalopathy Prediction Model (XGBoost, Voice-Based)

This repository contains a trained XGBoost model (`best_model.json`) designed to predict **covert hepatic encephalopathy (C-HE)** based on **voice data**, specifically sustained phonations of vowels **'a'** and **'e'**.

The model uses acoustic features extracted from these phonations and was trained using the low-level XGBoost API (`xgboost.Booster`) with a `DMatrix` data structure for optimized data handling.

---

## Model Details

- **Model type**: XGBoost (Gradient Boosted Trees)
- **Framework**: XGBoost low-level API (`xgboost.Booster`)
- **Model file**: `best_model.json` (saved via `model.save_model()`)
- **Input features**: Acoustic features extracted from vowels 'a' and 'e' — see below
- **Feature order**: Preserved inside the JSON model (`feature_names`)
- **Target variable**: Binary label indicating presence (1) or absence (0) of covert hepatic encephalopathy
- **Data format**: Assumes input data is a pandas DataFrame with a `labels` column and feature columns in the same order as used during training

---

## Usage Example

```python
import xgboost as xgb
import pandas as pd

# Load the trained model
model = xgb.Booster()
model.load_model("best_model.json")

# Load test data
test_data = pd.read_csv("your_test_data.csv")  # Replace with your test file path
y_test = test_data["labels"].values
X_test = test_data.drop(columns=["labels"])

# Prepare DMatrix with matching feature names
dtest = xgb.DMatrix(data=X_test, label=y_test, feature_names=X_test.columns.tolist())

# Predict
predictions = model.predict(dtest)

# Example: Threshold at 0.5 for binary classification
binary_preds = (predictions > 0.5).astype(int)
