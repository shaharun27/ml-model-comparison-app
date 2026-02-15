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
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Default Prediction",
    layout="wide"
)

st.title("Credit Card Default Prediction – ML Model Comparison")
st.markdown("BITS Pilani – Machine Learning Assignment 2")


# --------------------------------------------------
# DATA UTILITIES
# --------------------------------------------------

st.header("1️⃣ Dataset Utilities")

def generate_sample_data():
    np.random.seed(42)
    df = pd.DataFrame({
        "LIMIT_BAL": np.random.randint(10000, 100000, 200),
        "AGE": np.random.randint(21, 70, 200),
        "EDUCATION": np.random.choice(["Grad", "UG", "PhD"], 200),
        "BILL_AMT1": np.random.randint(0, 50000, 200),
        "PAY_AMT1": np.random.randint(0, 20000, 200),
        "target": np.random.choice([0, 1], 200)
    })
    return df

sample_df = generate_sample_data()

st.download_button(
    "📥 Download Sample Dataset",
    data=sample_df.to_csv(index=False),
    file_name="sample_credit_default.csv",
    mime="text/csv"
)

uploaded_file = st.file_uploader(
    "Upload Credit Card Dataset (CSV or XLS)",
    type=["csv", "xls"]
)

df = None

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, header=1)

        st.success("Dataset uploaded successfully!")

    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()


# --------------------------------------------------
# CONTINUE ONLY IF DATA EXISTS
# --------------------------------------------------

if df is not None:

    df.columns = df.columns.str.strip()

    if "default.payment.next.month" in df.columns:
        df.rename(columns={"default.payment.next.month": "target"}, inplace=True)

    if "default payment next month" in df.columns:
        df.rename(columns={"default payment next month": "target"}, inplace=True)

    if "target" not in df.columns:
        st.error("Dataset must contain a 'target' column.")
        st.stop()

    st.subheader("Original Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # DATA PREPROCESSING
    # --------------------------------------------------

    st.header("2️⃣ Data Preprocessing")

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # One-Hot Encoding
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    if categorical_cols:
        st.write("Categorical Columns Detected:", categorical_cols)
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        st.success("One-Hot Encoding Applied")

    # Show first 5 processed records
    st.subheader("Preview of Preprocessed Data (First 5 Rows)")
    st.dataframe(df.head())

    X = df.drop("target", axis=1)
    y = df["target"]

    st.write("Final Feature Shape:", X.shape)

    # --------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------

    st.header("3️⃣ Train-Test Split")

    test_size = st.slider("Select Test Size", 0.1, 0.4, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
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

        if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

        elif model_name == "Decision Tree":
            model = DecisionTreeClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

        elif model_name == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

        elif model_name == "Naive Bayes":
            model = GaussianNB()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

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
        # EVALUATION
        # --------------------------------------------------

        st.header("4️⃣ Model Evaluation")

        metrics_df = pd.DataFrame({
            "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
            "Value": [
                accuracy_score(y_test, y_pred),
                roc_auc_score(y_test, y_prob),
                precision_score(y_test, y_pred, zero_division=0),
                recall_score(y_test, y_pred, zero_division=0),
                f1_score(y_test, y_pred, zero_division=0),
                matthews_corrcoef(y_test, y_pred)
            ]
        })

        st.subheader("Overall Metrics")
        st.dataframe(metrics_df.set_index("Metric"))

        # Classification Report (Tabular)
        st.subheader("Classification Report")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df)

        # Confusion Matrix
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

else:
    st.info("Upload dataset to begin.")
