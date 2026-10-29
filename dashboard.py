
import requests
import pandas as pd
import streamlit as st

# ---------------- CONFIGURATION ----------------

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI Fraud Detection",
    layout="wide"
)

st.title("AI-Powered Financial Fraud Detection")
st.caption("Random Forest • FastAPI • MySQL")

# ---------------- API HELPERS ----------------

def get_history(limit=100):
    response = requests.get(
        f"{API_URL}/history",
        params={"limit": limit},
        timeout=10
    )
    response.raise_for_status()
    return response.json()["predictions"]


def predict_transaction(transaction):
    response = requests.post(
        f"{API_URL}/predict",
        json=transaction,
        timeout=30
    )
    response.raise_for_status()
    return response.json()


# ---------------- SIDEBAR ----------------

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    ["Overview", "Check Transaction", "Prediction History"]
)

# ---------------- OVERVIEW ----------------

if page == "Overview":
    st.header("Dashboard Overview")

    try:
        records = get_history()

        total = len(records)
        fraud_count = sum(
            row["predicted_label"] == 1 for row in records
        )
        legitimate_count = total - fraud_count

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Predictions", total)
        col2.metric("Fraud Alerts", fraud_count)
        col3.metric("Likely Legitimate", legitimate_count)

        if records:
            df = pd.DataFrame(records)

            st.subheader("Recent Predictions")
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("Prediction Distribution")
            st.bar_chart(
                pd.DataFrame({
                    "Category": [
                        "Fraud Alert",
                        "Likely Legitimate"
                    ],
                    "Count": [
                        fraud_count,
                        legitimate_count
                    ]
                }).set_index("Category")
            )
        else:
            st.info("No predictions recorded yet.")

    except requests.RequestException:
        st.error(
            "Cannot connect to the API. "
            "Start FastAPI in another terminal first."
        )

# ---------------- CHECK TRANSACTION ----------------

elif page == "Check Transaction":
    st.header("Check a Transaction")

    st.warning(
        "Enter the 15 numerical model features. "
        "These are anonymized dataset features, not raw card details."
    )

    feature_names = [
        "V14", "V10", "V4", "V12", "V17",
        "V3", "V11", "V16", "V2", "V9",
        "V7", "V21", "V20", "V19", "Amount"
    ]

    with st.form("transaction_form"):
        st.subheader("Transaction Features")

        values = {}

        columns = st.columns(3)

        for index, feature in enumerate(feature_names):
            with columns[index % 3]:
                default_value = 100.0 if feature == "Amount" else 0.0

                values[feature] = st.number_input(
                    feature,
                    value=default_value,
                    format="%.6f"
                )

        submitted = st.form_submit_button(
            "Analyze Transaction",
            type="primary",
            use_container_width=True
        )

    if submitted:
        try:
            with st.spinner("Analyzing transaction..."):
                result = predict_transaction(values)

            probability = float(result["fraud_probability"])
            label = int(result["predicted_label"])

            st.subheader("Prediction Result")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Fraud Probability",
                f"{probability:.6%}"
            )
            col2.metric(
                "Decision Threshold",
                f"{result['threshold']:.0%}"
            )
            col3.metric(
                "Prediction ID",
                result.get("prediction_id", "N/A")
            )

            if label == 1:
                st.error(
                    "Fraud Alert — manual review recommended."
                )
            else:
                st.success(
                    "Likely Legitimate — not a guarantee of safety."
                )

            st.caption(
                "A model prediction is an alert, not confirmation "
                "that a transaction is fraudulent."
            )

        except requests.RequestException as error:
            st.error(f"API request failed: {error}")

# ---------------- PREDICTION HISTORY ----------------

elif page == "Prediction History":
    st.header("Prediction History")

    if st.button("Refresh History"):
        st.rerun()

    try:
        records = get_history()

        if records:
            df = pd.DataFrame(records)

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            csv_data = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download History as CSV",
                data=csv_data,
                file_name="fraud_prediction_history.csv",
                mime="text/csv"
            )
        else:
            st.info("No prediction history available.")

    except requests.RequestException:
        st.error(
            "Cannot retrieve history. "
            "Check whether FastAPI is running."
        )
