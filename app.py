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
