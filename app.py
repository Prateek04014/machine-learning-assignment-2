import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

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


def main():

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

    try:
        X_proc, y_true = preprocess_input(df, preprocessor)
    except Exception as e:
        st.error(f"Error preprocessing uploaded data: {e}")
        st.stop()

    y_pred = model.predict(X_proc)

    st.subheader(f"Predictions — {model_choice}")

    if y_true is not None:  #
        st.subheader("Evaluation Metrics")

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_proc)[:, 1]
        else:
            y_proba = y_pred

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_proba),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }

        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value:.3f}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"],
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        with col2:
            st.subheader("Classification Report")
            report = classification_report(
                y_true, y_pred, target_names=["No Churn", "Churn"], output_dict=True
            )
            st.dataframe(pd.DataFrame(report).transpose().round(3))
    else:
        st.warning(
            "No 'Churn' column found in the uploaded file — showing predictions only. "
            "Include the true labels to see evaluation metrics and a confusion matrix."
        )

if __name__ == "__main__":
    main()
