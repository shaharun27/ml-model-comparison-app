# Credit Card Default Prediction – ML Assignment 2

**Problem Statement**

The objective of this project is to develop and evaluate multiple machine learning classification models to predict whether a credit card client will default on payment in the next month.

Credit default prediction is a critical financial risk management problem. By accurately identifying high-risk customers, financial institutions can:
* Reduce potential credit losses
* Improve loan approval decisions
* Optimize credit risk strategies
* Strengthen financial portfolio stability

This is a binary classification problem, where:
* **0** → No Default
* **1** → Default

The project compares six different machine learning models and deploys them using a Streamlit web application.

**Dataset Description**

* **Dataset:** Credit Card Default Dataset
* **Total Records:** 30,000
* **Features:** 23 input features + 1 target column

🔹 **Feature Categories**

The dataset contains financial, demographic, and repayment history information.

| Feature | Description |
| :--- | :--- |
| **LIMIT_BAL** | Credit limit amount |
| **SEX** | Gender (1 = Male, 2 = Female) |
| **EDUCATION** | Education level |
| **MARRIAGE** | Marital status |
| **AGE** | Age of the client |
| **PAY_0 to PAY_6** | Repayment status for previous months |
| **BILL_AMT1 to BILL_AMT6** | Bill amount for last 6 months |
| **PAY_AMT1 to PAY_AMT6** | Amount paid in last 6 months |

**Target Variable**

| Value | Meaning |
| :--- | :--- |
| 0 | No Default |
| 1 | Default |

**Models Implemented**

Six machine learning models were implemented and evaluated:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest
6. XGBoost

**Model Evaluation Results**

Based on the final trained models:

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9234 | 0.9643 | 0.9453 | 0.9641 | 0.9546 | 0.7115 |
| Decision Tree | 0.9950 | 0.9889 | 0.9960 | 0.9980 | 0.9970 | 0.9818 |
| KNN | 0.8793 | 0.8726 | 0.8965 | 0.9671 | 0.9305 | 0.4986 |
| Naive Bayes | 0.9309 | 0.9859 | 0.9440 | 0.9751 | 0.9593 | 0.7353 |
| Random Forest | 0.9908 | 0.9996 | 0.9911 | 0.9980 | 0.9945 | 0.9665 |
| XGBoost | 0.9975 | 1.0000 | 0.9980 | 0.9990 | 0.9985 | 0.9909 |



**Performance Analysis**

* **Logistic Regression**
    * Strong linear baseline model.
    * Good AUC (0.9643).
    * Slightly lower MCC compared to ensemble models.
    * Suitable when interpretability is required.
* **Decision Tree**
    * Extremely high accuracy (99.5%).
    * Very high F1 and MCC.
    * May slightly risk overfitting if not pruned properly.
    * Excellent performance overall.
* **K-Nearest Neighbors**
    * Moderate performance.
    * High recall but lower MCC.
    * Sensitive to scaling and feature distribution.
    * Computationally expensive for large datasets.
* **Naive Bayes**
    * Strong AUC (0.9859).
    * Fast training time.
    * Assumes feature independence.
    * Performs surprisingly well on this dataset.
* **Random Forest (Ensemble)**
    * Very high accuracy (99.08%).
    * Excellent AUC (0.9996).
    * Low variance and strong generalization.
    * Robust and stable performance.
* **XGBoost (Best Model)**
    * Highest accuracy: 99.75%.
    * Perfect AUC: 1.000.
    * Highest MCC: 0.9909.
    * Extremely low misclassification.
    * Captures complex non-linear relationships.
    * Best overall performing model.

**Final Model Ranking**

1. XGBoost – Best overall model
2. Decision Tree
3. Random Forest
4. Naive Bayes
5. Logistic Regression
6. KNN

**Streamlit Web Application Features**

The deployed application includes:
* CSV/XLS file upload
* Automatic target column detection
* Model selection from sidebar
* Train model button
* Accuracy, AUC, Precision, Recall, F1, MCC display
* Confusion Matrix visualization
* Classification Report table
* Download sample dataset option

**Repository Structure**

```text
ml-model-comparison-app/
│
├── app.py
├── README.md
├── requirements.txt
│
├── model/
│   ├── 1)Logistic_Regression.ipynb
│   ├── 2)decision_tree_model.ipynb
│   ├── 3)knn_model.ipynb
│   ├── 4)naive_bayes_model.ipynb
│   ├── 5)random_forest_model.ipynb
│   └── 6)xgboost_model.ipynb
│
└── data/
    ├── credit_default.csv
    └── sample_credit_default_test_data.csv

************************************************************************************************
**Key Insights**

Tree-based and ensemble models significantly outperform linear models.

XGBoost captures complex repayment behavior patterns effectively.

Credit default prediction benefits strongly from non-linear learning.

MCC confirms strong balanced classification performance.

Ensemble methods are highly suitable for financial risk modeling.


*************************************************************************************************

**Conclusion**

This project demonstrates that ensemble models, especially XGBoost, provide exceptional predictive performance for credit default prediction.

The implementation showcases:

Model comparison

Risk evaluation

Financial classification modeling

End-to-end deployment using Streamlit

This system can assist financial institutions in building automated, data-driven credit risk assessment pipelines.	