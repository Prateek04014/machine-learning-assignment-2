import streamlit as st
import pandas as pd
import joblib
import os

MODEL_DIR = "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}


@st.cache_resource
def load_preprocessor():
    return joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))

@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))


st.set_page_config(page_title="Telco Churn Classifier Demo", layout="wide")

st.title("Telco Customer Churn — Model Comparison App")

st.sidebar.header("Controls")
uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
model_choice = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

if uploaded_file is None:
    st.info("Upload a test CSV to get started (e.g. test_data.csv from this repo).")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("Preview of uploaded data")
st.dataframe(df.head())