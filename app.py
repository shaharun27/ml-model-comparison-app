import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Default Prediction",
    layout="wide"
)

st.title("Credit Card Default Prediction – ML Model Comparison")
st.markdown("BITS Pilani – Machine Learning Assignment 2")

def generate_sample_data():
    np.random.seed(42)

    start_id = 23999
    end_id = 30000
    n_rows = end_id - start_id + 1   # 6002 rows

    df = pd.DataFrame({
        "ID": np.arange(start_id, end_id + 1),
        "LIMIT_BAL": np.random.randint(10000, 1000000, n_rows),
        "SEX": np.random.choice([1, 2], n_rows),
        "EDUCATION": np.random.choice([1, 2, 3, 4], n_rows),
        "MARRIAGE": np.random.choice([1, 2, 3], n_rows),
        "AGE": np.random.randint(21, 75, n_rows),

        "PAY_0": np.random.randint(-2, 9, n_rows),
        "PAY_2": np.random.randint(-2, 9, n_rows),
        "PAY_3": np.random.randint(-2, 9, n_rows),
        "PAY_4": np.random.randint(-2, 9, n_rows),
        "PAY_5": np.random.randint(-2, 9, n_rows),
        "PAY_6": np.random.randint(-2, 9, n_rows),

        "BILL_AMT1": np.random.randint(0, 200000, n_rows),
        "BILL_AMT2": np.random.randint(0, 200000, n_rows),
        "BILL_AMT3": np.random.randint(0, 200000, n_rows),
        "BILL_AMT4": np.random.randint(0, 200000, n_rows),
        "BILL_AMT5": np.random.randint(0, 200000, n_rows),
        "BILL_AMT6": np.random.randint(0, 200000, n_rows),

        "PAY_AMT1": np.random.randint(0, 50000, n_rows),
        "PAY_AMT2": np.random.randint(0, 50000, n_rows),
        "PAY_AMT3": np.random.randint(0, 50000, n_rows),
        "PAY_AMT4": np.random.randint(0, 50000, n_rows),
        "PAY_AMT5": np.random.randint(0, 50000, n_rows),
        "PAY_AMT6": np.random.randint(0, 50000, n_rows),

        "target": np.random.choice([0, 1], n_rows)
    })

    return df


sample_df = generate_sample_data()

st.download_button(
    "📥 Download Sample Dataset",
    data=sample_df.to_csv(index=False),
    file_name="sample_credit_default.csv",
    mime="text/csv"
)

# --------------------------------------------------
# Upload Dataset
# --------------------------------------------------

st.header("Upload Credit Card Default Dataset (CSV or XLS)")
uploaded_file = st.file_uploader(
    "Upload file (must contain default column)",
    type=["csv", "xls"]
)

if uploaded_file is not None:

    # --------------------------------------------------
    # Read file (CSV or XLS)
    # --------------------------------------------------
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, header=1)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Clean column names
    df.columns = df.columns.str.strip()

    # Rename target column automatically
    if "default.payment.next.month" in df.columns:
        df.rename(columns={"default.payment.next.month": "target"}, inplace=True)

    if "default payment next month" in df.columns:
        df.rename(columns={"default payment next month": "target"}, inplace=True)

    if "target" not in df.columns:
        st.error("Dataset must contain a 'target' column.")
        st.stop()

    st.success("Dataset loaded successfully!")
    st.dataframe(df.head())

    # --------------------------------------------------
    # Train-Test Split
    # --------------------------------------------------

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    st.write("Train Shape:", X_train.shape)
    st.write("Test Shape:", X_test.shape)

    # --------------------------------------------------
    # Model Selection
    # --------------------------------------------------

    st.sidebar.header("Select Model")

    model_name = st.sidebar.selectbox(
        "Choose Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]
    )

    if st.button("Train Model"):

        model = None

        # --------------------------------------------------
        # Logistic Regression
        # --------------------------------------------------
        if model_name == "Logistic Regression":

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # --------------------------------------------------
        # Decision Tree
        # --------------------------------------------------
        elif model_name == "Decision Tree":

            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        # --------------------------------------------------
        # KNN
        # --------------------------------------------------
        elif model_name == "KNN":

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # --------------------------------------------------
        # Naive Bayes
        # --------------------------------------------------
        elif model_name == "Naive Bayes":

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = GaussianNB()
            model.fit(X_train_scaled, y_train)

            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # --------------------------------------------------
        # Random Forest
        # --------------------------------------------------
        elif model_name == "Random Forest":

            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        # --------------------------------------------------
        # XGBoost
        # --------------------------------------------------
        elif model_name == "XGBoost":

            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        # --------------------------------------------------
        # Evaluation
        # --------------------------------------------------

        st.header("Model Evaluation")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)

        metrics_df = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "AUC Score",
                "Precision",
                "Recall",
                "F1 Score",
                "MCC"
            ],
            "Value": [
                accuracy,
                auc,
                precision,
                recall,
                f1,
                mcc
            ]
        })

        st.dataframe(metrics_df.set_index("Metric"))

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="viridis",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"]
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        # --------------------------------------------------
        # Classification Report
        # --------------------------------------------------

        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred)
        st.text(report)

else:
    st.info("Upload dataset to begin.")
