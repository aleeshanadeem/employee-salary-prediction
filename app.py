import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.set_page_config(page_title="Employee Salary Prediction", page_icon="💼")

st.title("💼 Employee Salary Prediction")
st.write("Predict employee salary using Machine Learning.")

st.sidebar.header("Employee Information")

# ---------------------------
# User Inputs
# -----------------------------
job_title = st.sidebar.text_input("Job Title", "Machine Learning Engineer")

experience = st.sidebar.slider(
    "Years of Experience",
    0,
    30,
    5
)

education = st.sidebar.selectbox(
    "Education Level",
    ["High School", "Bachelor", "Master", "PhD"]
)

skills = st.sidebar.slider(
    "Skills Count",
    1,
    20,
    8
)

industry = st.sidebar.text_input(
    "Industry",
    "Technology"
)

company = st.sidebar.selectbox(
    "Company Size",
    ["Small", "Medium", "Large"]
)

location = st.sidebar.text_input(
    "Location",
    "USA"
)

remote = st.sidebar.selectbox(
    "Remote Work",
    ["Yes", "No"]
)

certifications = st.sidebar.slider(
    "Certifications",
    0,
    10,
    2
)

# -----------------------------
# Create DataFrame
# -----------------------------
input_df = pd.DataFrame({
    "job_title":[job_title],
    "experience_years":[experience],
    "education_level":[education],
    "skills_count":[skills],
    "industry":[industry],
    "company_size":[company],
    "location":[location],
    "remote_work":[remote],
    "certifications":[certifications]
})

# One Hot Encoding
input_df = pd.get_dummies(input_df)

# Match training columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Salary"):

    prediction = model.predict(input_df)

    st.success(f"Predicted Salary : ${prediction[0]:,.2f}")
