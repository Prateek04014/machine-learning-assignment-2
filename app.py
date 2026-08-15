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


def preprocess_input(df, preprocessor): #cleaning+transformation of data
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    y_true = None
    if "Churn" in df.columns: #handle both yes-no & 1-0 lables
        if df["Churn"].dtype == object:
            y_true = df["Churn"].map({"Yes": 1, "No": 0})
        else:
            y_true = df["Churn"]
        df = df.drop(columns=["Churn"])

    X_proc = preprocessor.transform(df)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    return X_proc, y_true



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

preprocessor = load_preprocessor()
model = load_model(MODEL_FILES[model_choice])

X_proc, y_true = preprocess_input(df, preprocessor)
y_pred = model.predict(X_proc)

st.subheader(f"Predictions — {model_choice}")
st.write(y_pred[:20])  # temporary check