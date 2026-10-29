# AI-Powered Financial Fraud Detection System

A machine learning project that estimates credit card transaction fraud risk using a Random Forest classifier, with a FastAPI prediction service, MySQL prediction history, and a Streamlit analytics dashboard.

## Features

- Exploratory data analysis of an imbalanced transaction dataset
- Random Forest classification
- Fraud probability and configurable decision threshold
- FastAPI endpoints for prediction and history
- MySQL storage of prediction results
- Streamlit dashboard with interactive Plotly charts
- Transaction history and CSV export

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- FastAPI
- Streamlit
- Plotly
- MySQL
- Joblib
- uv

## Project Structure

```text
.
├── main.py
├── dashboard.py
├── src/
│   ├── predict.py
│   └── database.py
├── models/
├── Notebooks/
├── requirements.txt
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "AI-Powered Financial Fraud Detection System"
```

### 2. Install dependencies

Install uv if it is not already available, then run:

```bash
uv sync
```

### 3. Configure environment variables

Create a local `.env` file with your MySQL connection settings. Never commit this file or share its password.

### 4. Prepare the database

Create the `fraud_detection` database and the required prediction-history table using the SQL schema in your project documentation.

### 5. Start the API

```bash
uv run uvicorn main:app --reload
```

### 6. Start the dashboard

In a second terminal:

```bash
uv run streamlit run dashboard.py
```

Open the local URL printed by Streamlit.

## Dataset

This project uses the Kaggle Credit Card Fraud Detection dataset. The model features include anonymized PCA-transformed variables and transaction amount.

The dataset is not included in this repository. Obtain it from the original dataset source and follow its applicable terms.

## Limitations

- Predictions are estimates, not definitive confirmation of fraud.
- The dataset's anonymized features are not directly interpretable as customer or merchant information.
- Model performance should be evaluated using held-out data and appropriate metrics for imbalanced classification.
- The application is an educational project, not a production banking fraud-prevention service.

## Future Improvements

- Model evaluation and threshold tuning on a dedicated validation set
- Improved monitoring and explainability
- Automated tests and deployment
- More comprehensive risk-analysis reporting

## Author

Manoj M