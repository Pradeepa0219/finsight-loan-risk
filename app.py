import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def load_model():
    csv_path = os.path.join(os.path.dirname(__file__), 'train.csv.csv')
    df = pd.read_csv(csv_path)
    
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
    df['IncomeToLoan'] = df['TotalIncome'] / df['LoanAmount']

    le = LabelEncoder()
    for col in ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']:
        df[col] = le.fit_transform(df[col])

    X = df.drop(['Loan_ID','Loan_Status'], axis=1)
    y = df['Loan_Status']

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)
    return model

model = load_model()

st.set_page_config(page_title="FinSight - Loan Predictor", page_icon="🏦", layout="centered")
st.title("🏦 FinSight — AI Loan Risk Predictor")
st.markdown("Fill in the applicant details below to predict loan approval.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)

with col2:
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=100)
    loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 60, 480, 300, 240, 84, 12])
    credit_history = st.selectbox("Credit History", [1.0, 0.0],
                                   format_func=lambda x: "Good (1.0)" if x == 1.0 else "Bad (0.0)")
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.divider()

if st.button("🔍 Predict Loan Approval", use_container_width=True):
    gender_enc = 1 if gender == "Male" else 0
    married_enc = 1 if married == "Yes" else 0
    dependents_enc = 3 if dependents == "3+" else int(dependents)
    education_enc = 0 if education == "Graduate" else 1
    self_emp_enc = 1 if self_employed == "Yes" else 0
    property_enc = 2 if property_area == "Urban" else (1 if property_area == "Semiurban" else 0)

    total_income = applicant_income + coapplicant_income
    emi = loan_amount / loan_term if loan_term > 0 else 0
    income_to_loan = total_income / loan_amount if loan_amount > 0 else 0

    input_data = pd.DataFrame([[
        gender_enc, married_enc, dependents_enc, education_enc,
        self_emp_enc, applicant_income, coapplicant_income,
        loan_amount, loan_term, credit_history, property_enc,
        total_income, emi, income_to_loan
    ]], columns=[
        'Gender','Married','Dependents','Education',
        'Self_Employed','ApplicantIncome','CoapplicantIncome',
        'LoanAmount','Loan_Amount_Term','Credit_History',
        'Property_Area','TotalIncome','EMI','IncomeToLoan'
    ])

    result = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.divider()
    if result == 1:
        st.success("✅ LOAN APPROVED!")
        st.metric("Approval Confidence", f"{round(probability[1]*100, 1)}%")
    else:
        st.error("❌ LOAN REJECTED")
        st.metric("Rejection Confidence", f"{round(probability[0]*100, 1)}%")

    st.subheader("📊 Input Summary")
    summary = pd.DataFrame({
        'Feature': ['Total Income','Loan Amount','EMI','Income to Loan Ratio','Credit History'],
        'Value': [total_income, loan_amount, round(emi,3), round(income_to_loan,2), credit_history]
    })
    st.write(summary)
