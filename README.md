# Credit Scoring — Machine Learning on High-Dimensional Banking Data

## Overview
This project focuses on building a machine learning model for **credit risk assessment**, a core problem in the banking and financial industry.  
The goal is to predict the probability of customer default based on structured customer and transaction-related data.

The project was completed as part of an intensive Data Science bootcamp and follows an end-to-end machine learning workflow.

---

## Dataset
- **Training data:** 175,000 samples × 61 features  
- **Test data:** 75,000 samples × 60 features  
- **Target variable:** Credit default indicator (binary classification)

The dataset is **high-dimensional** and includes a wide range of numerical and categorical features derived from customer behavior and aggregated transaction histories.

---

## Approach
The following steps were implemented:

- Exploratory data analysis and feature inspection
- Data preprocessing, including:
  - Handling missing values
  - Encoding categorical variables
  - Feature scaling
- Model training on tabular data
- Hyperparameter optimization using `RandomizedSearchCV`
- Model evaluation using classification metrics suitable for imbalanced data

A gradient boosting approach was selected due to its strong performance on large, structured datasets.

---

## Model
- **Algorithm:** LightGBM (Gradient Boosting)
- **Reasoning:** Efficient handling of high-dimensional tabular data and complex feature interactions
- **Evaluation:** Performance assessed using standard classification metrics, with emphasis on ROC-AUC

---

## Results
- **Validation ROC-AUC:** ~0.71

The model demonstrates solid discriminatory power and generalizes well to unseen data.

---

## Tools & Libraries
- Python
- pandas, NumPy
- scikit-learn
- LightGBM
- matplotlib

---

## Key Takeaways
- Gradient boosting models are highly effective for credit scoring tasks
- Feature engineering and proper preprocessing are critical in high-dimensional datasets
- Hyperparameter tuning significantly improves model performance

This project demonstrates practical experience with real-world financial data and production-relevant machine learning pipelines.
