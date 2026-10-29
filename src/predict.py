
from pathlib import Path
import joblib
import pandas as pd

# Locate saved models relative to this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

model = joblib.load(MODELS_DIR / "fraud_model.joblib")
features = joblib.load(MODELS_DIR / "fraud_features.joblib")
threshold_config = joblib.load(MODELS_DIR / "fraud_threshold.joblib")

THRESHOLD = threshold_config["threshold"]


def predict_transaction(transaction: dict) -> dict:
    """
    Predict whether a transaction should be flagged for fraud review.

    transaction must contain all selected model features.
    """
    missing = [feature for feature in features if feature not in transaction]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Keep the exact feature order used during training
    input_df = pd.DataFrame(
        [[transaction[feature] for feature in features]],
        columns=features
    )

    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(probability >= THRESHOLD)

    return {
        "fraud_probability": round(probability, 6),
        "threshold": THRESHOLD,
        "predicted_label": prediction,
        "result": (
            "Fraud alert - review required"
            if prediction == 1
            else "Likely legitimate"
        )
    }
