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
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Default Prediction",
    layout="wide"
)

st.title("Credit Card Default Prediction – ML Model Comparison")
st.markdown("BITS Pilani – Machine Learning Assignment 2")

# --------------------------------------------------
# SAMPLE DATA GENERATOR (REALISTIC PATTERN)
# --------------------------------------------------

def generate_sample_data():
    np.random.seed(42)

    start_id = 23999
    end_id = 30000
    n_rows = end_id - start_id + 1

    LIMIT_BAL = np.random.randint(10000, 1000000, n_rows)
    AGE = np.random.randint(21, 75, n_rows)

    PAY_0 = np.random.randint(-2, 9, n_rows)
    PAY_2 = np.random.randint(-2, 9, n_rows)

    BILL_AMT1 = np.random.randint(0, 200000, n_rows)
    PAY_AMT1 = np.random.randint(0, 50000, n_rows)

    # Realistic default logic
    target = (
        (PAY_0 > 2) |
        (PAY_2 > 2) |
        (PAY_AMT1 < BILL_AMT1 * 0.1)
    ).astype(int)

    df = pd.DataFrame({
        "ID": np.arange(start_id, end_id + 1),
        "LIMIT_BAL": LIMIT_BAL,
        "SEX": np.random.choice([1, 2], n_rows),
        "EDUCATION": np.random.choice([1, 2, 3, 4], n_rows),
        "MARRIAGE": np.random.choice([1, 2, 3], n_rows),
        "AGE": AGE,
        "PAY_0": PAY_0,
        "PAY_2": PAY_2,
        "PAY_3": np.random.randint(-2, 9, n_rows),
        "PAY_4": np.random.randint(-2, 9, n_rows),
        "PAY_5": np.random.randint(-2, 9, n_rows),
        "PAY_6": np.random.randint(-2, 9, n_rows),
        "BILL_AMT1": BILL_AMT1,
        "BILL_AMT2": np.random.randint(0, 200000, n_rows),
        "BILL_AMT3": np.random.randint(0, 200000, n_rows),
        "BILL_AMT4": np.random.randint(0, 200000, n_rows),
        "BILL_AMT5": np.random.randint(0, 200000, n_rows),
        "BILL_AMT6": np.random.randint(0, 200000, n_rows),
        "PAY_AMT1": PAY_AMT1,
        "PAY_AMT2": np.random.randint(0, 50000, n_rows),
        "PAY_AMT3": np.random.randint(0, 50000, n_rows),
        "PAY_AMT4": np.random.randint(0, 50000, n_rows),
        "PAY_AMT5": np.random.randint(0, 50000, n_rows),
        "PAY_AMT6": np.random.randint(0, 50000, n_rows),
        "target": target
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
# FILE UPLOAD
# --------------------------------------------------

st.header("Upload Credit Card Default Dataset (CSV or XLS)")
uploaded_file = st.file_uploader(
    "Upload file (must contain target column)",
    type=["csv", "xls"]
)

if uploaded_file is not None:

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    df.columns = df.columns.str.strip()

    # Auto rename possible target names
    rename_dict = {
        "default.payment.next.month": "target",
        "default payment next month": "target",
        "default": "target"
    }

    df.rename(columns=rename_dict, inplace=True)

    if "target" not in df.columns:
        st.error("Dataset must contain a 'target' column.")
        st.stop()

    # Remove ID column
    if "ID" in df.columns:
        df = df.drop("ID", axis=1)

    st.success("Dataset Loaded Successfully!")
    st.dataframe(df.head())

    # --------------------------------------------------
    # SPLIT DATA
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
    # MODEL SELECTION
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

        # Logistic / KNN / NB need scaling
        if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Initialize model
        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)

        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)

        elif model_name == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)

        elif model_name == "Naive Bayes":
            model = GaussianNB()

        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        elif model_name == "XGBoost":
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42
            )

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # --------------------------------------------------
        # EVALUATION
        # --------------------------------------------------

        st.header("Model Evaluation")

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC"],
            "Value": [accuracy, auc, precision, recall, f1, mcc]
        })

        st.dataframe(metrics_df.set_index("Metric"))

        # --------------------------------------------------
        # CONFUSION MATRIX
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
        # CLASSIFICATION REPORT
        # --------------------------------------------------

        st.subheader("Classification Report")

        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df)

else:
    st.info("Upload dataset to begin.")
