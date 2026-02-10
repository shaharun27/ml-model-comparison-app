# ML Assignment 2 – Credit Card Default Prediction

## Problem Statement
The objective of this project is to build and compare multiple machine learning
classification models to predict whether a credit card customer will default on
payment in the next month. The project demonstrates an end-to-end machine learning
workflow including data loading, model training, evaluation using multiple metrics,
and deployment through an interactive Streamlit web application.

This project is developed as part of **Machine Learning – Assignment 2** for the  
**M.Tech (AIML / DSE) Work Integrated Learning Programme, BITS Pilani**.

---

## Dataset Description  (1 Mark)

- **Dataset Name:** Default of Credit Card Clients
- **Source:** UCI Machine Learning Repository
- **Link:** https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

### Dataset Details
The dataset contains information about credit card holders in Taiwan, including
demographic data, credit limits, repayment history, bill amounts, and previous
payments. The task is to predict whether a customer will default on payment in the
next month.

- **Total Instances:** 30,000
- **Number of Features:** 23
- **Target Variable:** `default.payment.next.month` (renamed to `target`)
  - `0` → No Default
  - `1` → Default
- **Missing Values:** None

The dataset satisfies the assignment requirement of a minimum of **12 features**
and **500 instances**.

---

## Models Used and Evaluation Metrics  (6 Marks)

The following six classification models are implemented using the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

Each model is evaluated using the following metrics:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Decision Tree | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| KNN | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Naive Bayes | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Random Forest (Ensemble) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| XGBoost (Ensemble) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

*All metrics are computed dynamically inside the Streamlit application.*

---

## Observations on Model Performance  (3 Marks)

| ML Model Name | Observation |
|--------------|-------------|
| Logistic Regression | Provides a strong baseline and is easy to interpret, but struggles with complex non-linear patterns in credit data. |
| Decision Tree | Captures non-linear relationships well but may overfit if not controlled. |
| KNN | Performs reasonably well but is sensitive to feature scaling and computationally expensive for large datasets. |
| Naive Bayes | Fast and simple, but its assumption of feature independence does not fully hold for credit data. |
| Random Forest (Ensemble) | Shows strong performance by reducing overfitting through ensemble learning. |
| XGBoost (Ensemble) | Achieves the best overall performance by modeling complex feature interactions and correcting errors iteratively. |

---

## Project Structure

