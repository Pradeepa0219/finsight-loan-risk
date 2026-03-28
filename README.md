# 🏦 FinSight — AI Loan Risk Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.7-blue)
![Model](https://img.shields.io/badge/Model-GradientBoosting-green)
![Accuracy](https://img.shields.io/badge/Accuracy-80.49%25-brightgreen)

## 📌 Problem Statement
Banks manually review thousands of loan applications.
FinSight automates this using ML with explainable predictions.

## 🧠 Models Compared
| Model | Accuracy |
|---|---|
| Gradient Boosting | 80.49% |
| Random Forest | 79.67% |
| Logistic Regression | 78.86% |
| XGBoost | 76.42% |
| Decision Tree | 66.67% |

## 💡 Key Features
- Exploratory Data Analysis (EDA)
- Feature Engineering (EMI, TotalIncome, IncomeToLoan ratio)
- 5 ML Models benchmarked
- SHAP Explainability — Credit History = #1 factor
- Interactive Streamlit dashboard

## 🛠️ Tech Stack
Python, Pandas, Scikit-learn, XGBoost, SHAP, Streamlit

## ▶️ How to Run
pip install -r requirements.txt
streamlit run app.py
