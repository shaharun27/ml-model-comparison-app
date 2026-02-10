import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# --- 1. MODEL IMPORTS ---
try:
    from model.logistic_regression import train_and_evaluate as lr_eval
    from model.decision_tree_model import train_and_evaluate as dt_eval
    from model.knn_model import train_and_evaluate as knn_eval
    from model.naive_bayes_model import train_and_evaluate as nb_eval
    from model.random_forest_model import train_and_evaluate as rf_eval
    from model.xgboost_model import train_and_evaluate as xgb_eval
except ImportError as e:
    st.error(f"❌ Module Import Error: {e}")

st.set_page_config(page_title="BITS Assignment 2", layout="wide")
st.title("💳 Credit Card Default Prediction Dashboard")

# --- 2. DATA LOADING ---
uploaded_file = st.file_uploader("Upload Dataset (credit_default.csv/xls)", type=["csv", "xls"])

if uploaded_file:
    try:
        # header=1 skips the title row and starts at the actual column names
        df = pd.read_excel(uploaded_file, header=1, engine='xlrd')
        
        # Clean column names (remove spaces and special characters)
        df.columns = [str(c).strip() for c in df.columns]
        
        # SMART TARGET DETECTION
        # This looks for common variations of the target column name
        target_options = ['default.payment.next.month', 'default payment next month', 'TARGET', 'target']
        found_target = None
        for col in df.columns:
            if col.lower() in [t.lower() for t in target_options]:
                found_target = col
                break
        
        if found_target:
            df.rename(columns={found_target: "target"}, inplace=True)
            st.success(f"✅ Target found: '{found_target}'")
            st.write("### Dataset Preview", df.head())

            # --- 3. TRAINING & EVALUATION ---
            X = df.drop(columns=['target'])
            y = df['target']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_choice = st.selectbox("Select Model", 
                ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"])

            mapping = {
                "Logistic Regression": lr_eval, "Decision Tree": dt_eval, "KNN": knn_eval,
                "Naive Bayes": nb_eval, "Random Forest": rf_eval, "XGBoost": xgb_eval
            }

            if st.button("🚀 Run Evaluation"):
                res = mapping[model_choice](X_train, X_test, y_train, y_test)
                
                # Mandatory 6 Metrics [Requirement: 6 Marks]
                st.subheader(f"📊 {model_choice} Metrics")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{res['Accuracy']:.4f}")
                c2.metric("AUC Score", f"{res['AUC']:.4f}")
                c3.metric("Precision", f"{res['Precision']:.4f}")
                c1.metric("Recall", f"{res['Recall']:.4f}")
                c2.metric("F1 Score", f"{res['F1']:.4f}")
                c3.metric("MCC Score", f"{res['MCC']:.4f}")

                # Confusion Matrix [Requirement: 1 Mark]
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(confusion_matrix(y_test, res['y_pred']), annot=True, fmt='d', cmap='RdPu')
                st.pyplot(fig)
        else:
            st.error("❌ Target column not found. Available columns: " + ", ".join(df.columns[:5]) + "...")
            st.info("Check if Row 2 (index 1) of your file actually contains the column names.")

    except Exception as e:
        st.error(f"🛑 Error loading file: {e}")