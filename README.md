# Telecom Customer Churn Prediction

## Overview

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn. The project focuses on building a reliable classification pipeline, comparing different model families, tuning ensemble models, selecting an operating threshold, and translating model results into business actions.

## Business Problem

Customer churn can reduce recurring revenue and increase the cost of acquiring replacement customers. The objective is to identify customers with a higher probability of churn so that a telecom company can prioritize retention efforts.

## Dataset

- **Dataset:** IBM Telco Customer Churn
- **Rows:** 7,043
- **Columns:** 21
- **Target:** `Churn`
- **Target classes:** `Yes` / `No`

The notebook loads the public CSV directly, so no dataset file is required in the repository.

## Project Workflow

```text
Business Problem
      ↓
Data Understanding
      ↓
Univariate + Bivariate EDA
      ↓
Data Cleaning
      ↓
Missing Value Handling
      ↓
Outlier Analysis
      ↓
Feature Engineering
      ↓
Categorical Encoding + Scaling
      ↓
Train/Test Split
      ↓
Baseline Model Comparison
      ↓
Cross-Validation
      ↓
Hyperparameter Tuning
      ↓
Final Model Selection
      ↓
Threshold Optimization
      ↓
Final Test Evaluation
      ↓
Feature Importance + Business Insights
```

## Exploratory Data Analysis

Important associations found in the dataset:

- Month-to-month customers had a **42.71% churn rate**, compared with **11.27%** for one-year contracts and **2.83%** for two-year contracts.
- Fiber-optic customers had a **41.89% churn rate**, compared with **18.96%** for DSL customers.
- Electronic-check customers had a **45.29% churn rate**, compared with **16.71%** for automatic bank-transfer customers.
- Customers without online security had a **41.77% churn rate**, compared with **14.61%** for customers with the service.
- Customers without technical support had a **41.64% churn rate**, compared with **15.17%** for customers with the service.

These are observed associations and should not be interpreted as causal effects.

## Data Preprocessing

- Converted `TotalCharges` from text to numeric.
- Identified **11 missing `TotalCharges` values (0.156%)**.
- Used median imputation for numerical variables inside the preprocessing pipeline.
- Used most-frequent imputation for categorical variables.
- Applied one-hot encoding to categorical variables.
- Applied standard scaling to numerical variables.
- Removed `customerID` from the model because it is an identifier rather than a predictive feature.
- Used a stratified train/test split to preserve class proportions.

## Feature Engineering

The project created a small set of business-oriented features:

- `TotalServices`
- `AvgMonthlySpend`
- `TenureGroup`
- `IsLongTermContract`

Outlier detection was performed using the IQR method. No selected numerical variable had IQR-based outliers requiring removal, so legitimate observations were retained.

## Models Compared

### Baseline / candidate models

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

### Ensemble learning

- Random Forest — bagging
- XGBoost — gradient boosting

## Model Selection

Because churn is the positive class of interest and the dataset is imbalanced, model comparison considered:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

**PR-AUC was used as the primary model-selection metric.**

The tuned XGBoost model achieved the highest training cross-validation PR-AUC among the tuned ensemble candidates:

**CV PR-AUC: 0.6701**

## Threshold Optimization

The default probability threshold of 0.50 was not assumed to be optimal.

Using out-of-fold predictions from the training data, the threshold was selected by maximizing F1.

**Selected threshold: 0.30**

This increases the model's sensitivity to potential churners and reflects the business cost of missing customers who may leave.

## Final Test Performance

The final model was selected before using the test set for final evaluation.

**Model:** Tuned XGBoost  
**Threshold:** 0.30

| Metric | Test Result |
|---|---:|
| Accuracy | **76.3%** |
| Precision | **53.7%** |
| Recall | **76.5%** |
| F1 | **63.1%** |
| ROC-AUC | **84.72%** |
| PR-AUC | **66.19%** |

### Interpretation

The final model identifies approximately **76.5% of actual churners** on the unseen test set. The lower threshold prioritizes recall, which can be useful when the business would rather contact additional customers than miss a potential churner.

## Key Predictive Signals

The XGBoost feature-importance analysis highlighted:

- Long-term contract status
- Internet service type
- Online security
- Technical support
- Contract type
- Tenure
- Payment method

Feature importance indicates predictive usefulness; it does not establish causality.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Google Colab

## Repository Contents

```text
Telecom-Customer-Churn/
│
├── Telecom_Customer_Churn.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

1. Open `Telecom_Customer_Churn.ipynb` in Google Colab.
2. Run all cells from top to bottom.
3. The notebook loads the public dataset automatically.
4. Review the EDA, model comparison, tuning, threshold optimization and final evaluation.

## Limitations

- The dataset is a historical public dataset and may not represent a current telecom customer population.
- EDA relationships are observational and do not imply causality.
- The selected threshold assumes that maximizing F1 is an appropriate operating objective; a real business deployment should incorporate the monetary costs of false positives and false negatives.

## Future Improvements

- Calibrate predicted probabilities.
- Incorporate customer-level cost/benefit information for threshold selection.
- Validate the model on a newer or external telecom dataset.
- Monitor model performance and churn distribution after deployment.
