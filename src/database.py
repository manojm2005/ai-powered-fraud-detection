
import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "fraud_detection")
    )


def save_prediction(result: dict) -> int:
    connection = get_connection()
    cursor = connection.cursor()

    try:
        query = """
            INSERT INTO transaction_predictions
            (fraud_probability, threshold_value, predicted_label, result)
            VALUES (%s, %s, %s, %s)
        """

        values = (
            result["fraud_probability"],
            result["threshold"],
            result["predicted_label"],
            result["result"]
        )

        cursor.execute(query, values)
        connection.commit()

        return cursor.lastrowid

    finally:
        cursor.close()
        connection.close()
