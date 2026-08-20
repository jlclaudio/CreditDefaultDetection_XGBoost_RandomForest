# Credit Card Default Prediction

## Overview
This project addresses the problem of predicting default payments of credit card clients in Taiwan [cite: 1]. The goal is to build and compare machine learning classifiers (Random Forest and XGBoost) to predict whether a client will default on their payment the next month [cite: 1, 2]. The project also explores feature reduction using Lasso regression, handles dataset imbalance, and interprets the model using SHAP [cite: 1, 2].

This was originally developed as part of a final project for the M.Sc. in High Performance Computing at Politecnico di Milano [cite: 1].

## Dataset
The dataset (`Dataset3.csv`) contains 30,000 entries and 25 variables collected from April 2005 to September 2005 [cite: 1, 2].
*   **Features:** Demographic factors (SEX, EDUCATION, MARRIAGE, AGE), historical repayment statuses (PAY_0 to PAY_6), bill statement amounts (BILL_AMT1 to BILL_AMT6), and previous payment amounts (PAY_AMT1 to PAY_AMT6) [cite: 1].
*   **Data Quality:** No missing values were found during exploratory data analysis [cite: 1].
*   **Imbalance:** The target variable (`default.payment.next.month`) is highly unbalanced, with a default rate of 22.12% (6,636 defaults vs. 23,364 non-defaults) [cite: 1, 2].

## Methodology
1.  **Baseline Classifiers:** Random Forest and XGBoost classifiers were implemented, with optimal hyperparameters determined via `GridSearchCV` [cite: 1, 2].
2.  **Feature Selection:** Lasso Regression (with L1 regularization) was used to reduce the feature space from 23 to 16 key variables while maintaining predictive power [cite: 1, 2]. Selected features include: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`, `PAY_0`, `PAY_2`, `PAY_3`, `PAY_4`, `PAY_6`, `BILL_AMT1`, `PAY_AMT1`, `PAY_AMT2`, `PAY_AMT4`, `PAY_AMT5`, and `PAY_AMT6` [cite: 1, 2].
3.  **Handling Imbalanced Data:**
    *   **SMOTE** (Synthetic Minority Over-sampling Technique) was used to generate synthetic examples for the minority class [cite: 1, 2].
    *   **Threshold Optimization** was applied using the Precision-Recall curve to find the optimal classification threshold (best threshold found: 0.5345) [cite: 1, 2].
4.  **Interpretability:** SHAP (SHapley Additive exPlanations) values were used to identify the top features driving default [cite: 1, 2].

## Key Results

### Model Performance Comparison
| Model | AUC | F1-Score |
| :--- | :--- | :--- |
| **XGB Baseline** | 0.7788 | 0.5352 |
| **XGB Lasso** | 0.7778 | 0.5340 |
| **XGBoost (Lasso) + Thresholding** | 0.7778 | 0.5426 |
| **RF Lasso** | 0.7751 | 0.5388 |
| **RF Baseline** | 0.7750 | 0.5429 |
| **XGBoost (Lasso) + SMOTE** | 0.7434 | 0.4224 |
*(Data extracted from the performance comparison table [cite: 1, 2])*

### Insights
*   **Best Overall Model:** The XGBoost Baseline achieved the best overall AUC (0.7788), while the Random Forest Baseline yielded the highest F1-score (0.5429) [cite: 1, 2].
*   **Dimensionality Reduction:** Lasso effectively reduced the features to 16 without a significant drop in AUC or F1 metrics, creating a simpler and more computationally efficient model [cite: 1, 2].
*   **Business Strategies:**
    *   For strict **risk aversion** (identifying maximum defaulters), SMOTE is recommended as it boosted minority class recall to 85% [cite: 1, 2].
    *   For **balanced performance**, Threshold Optimization (setting threshold = 0.5345) on the XGBoost Lasso model is the optimal strategy [cite: 1, 2].
*   **Drivers of Default:** SHAP analysis revealed that historical payment statuses (especially `PAY_0`) and credit limit (`LIMIT_BAL`) are the most critical indicators of default [cite: 1, 2].
