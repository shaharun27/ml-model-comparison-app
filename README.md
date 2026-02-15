## Credit Card Default Prediction – ML Assignment 2 

## Problem Statement

The objective of this project is to predict whether a credit card client will default on payment in the next month based on historical financial and demographic information.

Credit risk prediction is a critical problem for financial institutions. Accurate prediction helps banks:

Reduce financial losses

Improve credit approval strategies

Manage risk exposure

Optimize lending decisions

This is a binary classification problem, where:

0 → No Default

1 → Default

Six different Machine Learning models are implemented, compared, and deployed using a Streamlit web application.

------------------------------------------------------------------------
## Dataset Description

Dataset: UCI Credit Card Default Dataset
Source: UCI Machine Learning Repository

Dataset Characteristics

Number of Instances: 30,000

Number of Features: 23

Target Column: target

Missing Values: None

------------------------------------------------------------------------
## Feature Categories

Demographic Information

LIMIT_BAL

SEX

EDUCATION

MARRIAGE

AGE

Payment History (Past 6 Months)

PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6

Bill Statements

BILL_AMT1 to BILL_AMT6

Previous Payments

PAY_AMT1 to PAY_AMT6

------------------------------------------------------------------------

## Target Variable
Value	Meaning
0	No Default
1	Default

------------------------------------------------------------------------

## Models Implemented

The following six classification models were trained and evaluated:

Logistic Regression

Decision Tree Classifier

K-Nearest Neighbors (KNN)

Naive Bayes (Gaussian)

Random Forest (Ensemble)

XGBoost (Ensemble)

------------------------------------------------------------------------

## Model Performance (From Your Streamlit Results)
ML Model	          Accuracy	AUC	   Precision	Recall	F1 Score	MCC
Logistic Regression	  0.8218	0.7503	0.6961	    0.2795	0.3989	  0.3615
Decision Tree	      0.7452	0.6454	0.4110	    0.4724	0.4396	  0.2768
KNN	                  0.8068	0.7169	0.5714	    0.3465	0.4314	  0.3380
Naive Bayes	          0.5695	0.7795	0.3097	    0.8425	0.4529	  0.2799
Random Forest	      0.8351	0.7977	0.6842	    0.4094	0.5123	  0.4406
XGBoost             0.8310	    0.8026	0.6783	    0.3819	0.4887	  0.4203

------------------------------------------------------------------------

## Model Performance Observations
## Logistic Regression

Good overall accuracy (82.18%)

Moderate AUC (0.75)

Low recall (27.95%) → misses many defaulters

Balanced MCC (0.36)

Suitable baseline linear model

## Decision Tree

Lower accuracy (74.52%)

Weak AUC (0.64)

Better recall than Logistic Regression

Slightly unstable due to single tree structure

Prone to overfitting

## K-Nearest Neighbors (KNN)

Good accuracy (80.68%)

Moderate recall (34.65%)

Performs reasonably after scaling

Sensitive to feature scaling and dimensionality

## Naive Bayes

Lowest accuracy (56.95%)

Very high recall (84.25%)

Very low precision (30.97%)

Predicts too many customers as defaulters

High false positive rate

## Random Forest (Best Performing Model)

Highest accuracy (83.51%)

Strong AUC (0.7977)

Best F1 score (0.5123)

Highest MCC (0.4406)

Good balance between precision and recall

Best overall model for this dataset

## XGBoost

Very close to Random Forest

Highest AUC (0.8026)

Good overall performance

Slightly lower F1 than Random Forest

Strong ensemble model

------------------------------------------------------------------------

## Overall Ranking (Based on F1 & MCC)

1) Random Forest

2) XGBoost

3) Logistic Regression

4) KNN

5) Naive Bayes

6) Decision Tree

------------------------------------------------------------------------

## Streamlit Web Application Features

The deployed Streamlit app provides:

Upload CSV or Excel file

Automatic preprocessing

Model selection from sidebar

Evaluation metrics table

Confusion Matrix visualization

Classification Report

Download predictions as CSV

------------------------------------------------------------------------

## Repository Structure
ml-model-comparison-app/
│
├── app.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── artifacts/
│   ├── submission_csv/
│   ├── 1_logistic_regression.ipynb
│   ├── 2_decision_tree.ipynb
│   ├── 3_knn.ipynb
│   ├── 4_naive_bayes.ipynb
│   ├── 5_random_forest.ipynb
│   └── 6_xgboost.ipynb
│
└── data/
    ├── credit_default.xls
    └── credit_default_test_data.xls
ml-model-comparison-app/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── model/
│   ├── 1_logistic_regression.ipynb
│   ├── 2_decision_tree.ipynb
│   ├── 3_knn.ipynb
│   ├── 4_naive_bayes.ipynb
│   ├── 5_random_forest.ipynb
│   └── 6_xgboost.ipynb
│
└── data/
    ├── credit_default.xls
    └── credit_default_test_data.xls
    
------------------------------------------------------------------------
## Key Insights

Ensemble models perform best for credit risk problems.

Recall is very important in default prediction.

Random Forest gives best overall balance.

Naive Bayes detects defaulters well but produces too many false alarms.

XGBoost gives highest AUC.

------------------------------------------------------------------------
## Conclusion

Random Forest is the most suitable model for credit default prediction in this project, achieving the best balance across:

Accuracy

F1 Score

MCC

Generalization performance

XGBoost is a strong alternative with excellent AUC performance.