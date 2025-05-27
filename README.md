# 🛒 Retail Stock-Out Prediction System

## 📌 Project Overview

This project is a machine learning solution aimed at predicting **out-of-stock events** in a retail supply chain. Out-of-stock lead to missed sales opportunities, unhappy customers, and decreased revenue. By forecasting such events, businesses can optimize inventory planning and reduce operational risks.

## 💡 Problem Statement

Retail companies often experience **out-of-stock (OOS)** scenarios without clear visibility into the cause. This notebook builds a predictive model to determine whether an item is at risk of being out of stock in the future, based on operational, sales, and risk features.

## 🧠 Methodology

### 1. Data Overview & Preprocessing
- **Rows**: 192,994 unique purchase orders  
- **Features**: 23 columns including:
  - 16 `numeric` variables
  - 7 `categorical` variables (mostly binary)

### 2. Exploratory Data Analysis
- Performed univariate and bivariate analysis to understand feature behavior.
- Analyzed correlation between independent variables and dependent variable.

## 🔍 Key Features

| Feature Name          | Description                                  |
|-----------------------|----------------------------------------------|
| `po_nbr`              | Purchase order number                       |
| `present_inv`         | Current inventory level                     |
| `lead_time`           | Estimated restocking lead time              |
| `total_whpk_qty`      | Total quantity at warehouse or packaging    |
| `fcst_3_mo` - `fcst_9_mo` | Future demand forecast                |
| `sls_1_mo` - `sls_9_mo` | Historical sales                        |
| `minimum_stock_qty`   | Minimum stock threshold                     |
| `internal_risk1-3`    | Internal business risk indicators           |
| `identified_defect`   | Flag for product defects                    |
| `spare_parts_overdue` | Delays or backlog for required spare parts  |
| `avg_ratings_6_mo` / `avg_ratings_12_mo` | Average customer ratings         |
| `stock_overdue`       | Overdue stock count                         |
| `pdt_recsys`          | Flag from a product recommendation system (e.g., "recommended to stock this") |
| `out_of_stock`        | Binary target variable (1 = Yes, 0 = No)     |

### 3. Data Preprocessing
- Replaced negative ratings with median values or zeros based on context.
- Imputed missing values (e.g., `lead_time` with median).
- Encoded categorical variables and scaled numerical features.
- Treated outliers using Interquartile Range (IQR) and Log Transformation.

### 4. Imbalanced Classification
- Experimented with **Under Sampling**, **Over Sampling**, and **Hybrid** methods to balance the classes since `out_of_stock = 1` was the minority.
- Final choice: Class weights, random under sampling, random over sampling, Synthetic Minority Oversampling Technique (SMOTE), SMOTE + Tomek Links, Borderline SMOTE, and Adaptive Synthetic (ADASYN).

### 5. Model Building
- Compared multiple classifiers:
  - **Logistic Regression**
  - **Decision Tree**
  - **Random Forest**
  - **K-Nearest Neighbors**
  - **Naive Bayes**
  - **XGBoost**
  - **LightGBM**
- Used `stratified k-fold` cross-validation and `Area Under Precision-Recall Curve (AUPRC)`, `recall`, `precision`, and `F1-score` for evaluation.
- Performed **hyperparameter tuning** using `BayesianOptimization` for efficient parameter search.

### 6. Evaluation
- Best model: **XGBoost**, retrained with class weights and Bayesian optimization, achieving recall score of 73% and AUPRC score of 16% for out-of-stock predictions.
- Generated:
  - Classification Report
  - Confusion Matrix
  - Precision-Recall Curve

## 📈 Results

- XGBoost is chosen as the final prediction model, achieving a recall of approximately 0.73, F1 score of 0.13, and PRC AUC of 0.16 after tuning with class weights.
- The project provides actionable insights into preventing stock-outs before they occur.

## 📦 Future Improvements

- Perform advanced feature engineering by creating new useful columns to capture more complex relationships.
- Implement feature selection techniques (e.g., RFE, SHAP) to identify the impactful features.
- Integrate external data sources that could impact stock levels (e.g., public holidays, supplier performance metrics).
- Explore different ensemble methods (e.g., boosting, bagging, stacking).
- Experiment with deep learning neural networks.
- More advanced sampling techniques (e.g., NearMiss, Edited Nearest Neighbour).

## 🤝 Acknowledgements

This project was independently created as a showcase for solving real-world inventory challenges using machine learning.
