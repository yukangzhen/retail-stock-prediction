# 🛒 Retail Stock-Out Prediction System

## 📌 Project Overview

This project is a machine learning solution aimed at predicting **stock-out events** in a retail supply chain. Stock-outs lead to missed sales opportunities, unhappy customers, and decreased revenue. By forecasting such events, businesses can optimize inventory planning and reduce operational risks.

## 💡 Problem Statement

Retail companies often experience **out-of-stock (OOS)** scenarios without clear visibility into the cause. This notebook builds a predictive model to determine whether an item is at risk of being out of stock in the future, based on operational, sales, and risk features.

## 🧠 Methodology

### 1. Data Overview & Preprocessing
- **Rows**: 192,994 unique purchase orders  
- **Features**: 23 columns including:
  - `present_inv` (inventory),
  - `lead_time` (days to restock),
  - `fcst_3_mo` to `fcst_9_mo` (sales forecasts),
  - `sls_1_mo` to `sls_9_mo` (historical sales),
  - `internal_risk1-3`, `production_quality_risk`,
  - `avg_ratings_6_mo`, `avg_ratings_12_mo`,
  - `out_of_stock` (target)

### 2. Exploratory Data Analysis
- Identified skewed distributions and missing values (`lead_time` had ~6% missing).
- Performed univariate and bivariate analysis to understand feature behavior.
- Analyzed correlation between inventory metrics and the likelihood of stock-out.

### 3. Feature Engineering
- Imputed missing values.
- Encoded categorical variables and scaled numerical features.
- Created derived variables for risk and sentiment analysis.

### 4. Imbalanced Classification
- Applied **Random Over Sampling** to balance the classes since `out_of_stock = 1` was the minority.
- Visualized class distribution before and after resampling.

### 5. Model Building
- Compared multiple classifiers:
  - **Logistic Regression**
  - **Random Forest**
  - **LightGBM**
- Used cross-validation and AUC-ROC, recall, precision, and F1-score for evaluation.
- Performed **hyperparameter tuning** using `RandomizedSearchCV`.

### 6. Evaluation
- Best model: **LightGBM**, tuned to maximize recall for out-of-stock predictions.
- Generated:
  - Confusion Matrix
  - ROC Curve
  - Precision-Recall Curve
- Model successfully identified OOS risk factors and showed strong classification metrics after tuning.

## 🔍 Key Features

| Feature Name | Description |
|--------------|-------------|
| `present_inv` | Current inventory level |
| `lead_time` | Restocking lead time |
| `fcst_3_mo` - `fcst_9_mo` | Future demand forecast |
| `sls_1_mo` - `sls_9_mo` | Historical sales |
| `minimum_stock_qty` | Minimum stock threshold |
| `internal_risk1-3` | Internal business risk indicators |
| `identified_defect` | Flag for product defects |
| `avg_ratings_6_mo` / `12_mo` | Customer sentiment proxy |
| `out_of_stock` | Binary target variable |

## 🛠 Tech Stack

- **Python**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn**
- **LightGBM**
- **Imbalanced-learn**

## 📈 Results

- LightGBM achieved strong classification performance with improved recall for the positive class.
- The project provides actionable insights into preventing stock-outs before they occur.

## 📦 Future Improvements

- Deploy model into a real-time inventory management system.
- Integrate live sales and warehouse APIs.
- Implement SHAP values or other model explainability techniques.

## 🤝 Acknowledgements

This project was independently created as a showcase for solving real-world inventory challenges using machine learning.
