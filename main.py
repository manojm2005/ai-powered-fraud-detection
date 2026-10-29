
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.predict import predict_transaction
from src.database import get_connection, save_prediction

app = FastAPI(
    title="AI-Powered Financial Fraud Detection API",
    description="Fraud detection API with MySQL prediction history",
    version="1.0.0"
)


# Transaction input: 15 features selected during model training
class Transaction(BaseModel):
    V14: float
    V10: float
    V4: float
    V12: float
    V17: float
    V3: float
    V11: float
    V16: float
    V2: float
    V9: float
    V7: float
    V21: float
    V20: float
    V19: float
    Amount: float


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running"
    }


# Predict fraud and save the result in MySQL
@app.post("/predict")
def predict(transaction: Transaction):
    try:
        result = predict_transaction(transaction.model_dump())
        prediction_id = save_prediction(result)

        return {
            "prediction_id": prediction_id,
            **result
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction or database operation failed"
        )


# Retrieve recent prediction history from MySQL
@app.get("/history")
def get_prediction_history(
    limit: int = Query(default=20, ge=1, le=100)
):
    connection = None
    cursor = None

    try:
        connection = get_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            """
            SELECT
                id,
                fraud_probability,
                threshold_value,
                predicted_label,
                result,
                created_at
            FROM transaction_predictions
            ORDER BY id DESC
            LIMIT %s
            """,
            (limit,)
        )

        records = cursor.fetchall()

        for record in records:
            record["fraud_probability"] = float(
                record["fraud_probability"]
            )
            record["threshold_value"] = float(
                record["threshold_value"]
            )
            record["predicted_label"] = int(
                record["predicted_label"]
            )

            if record["created_at"] is not None:
                record["created_at"] = (
                    record["created_at"].isoformat()
                )

        return {
            "count": len(records),
            "predictions": records
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve prediction history"
        )

    finally:
        if cursor is not None:
            cursor.close()

        if connection is not None and connection.is_connected():
            connection.close()
