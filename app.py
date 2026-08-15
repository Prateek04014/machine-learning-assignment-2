import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests #live data pull button

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
TEST_DATA_URL = "https://raw.githubusercontent.com/Prateek04014/machine-learning-assignment-2/main/test_data.csv"


@st.cache_resource
def load_preprocessor():
    return joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))

@st.cache_resource
def load_model(filename):
    return joblib.load(os.path.join(MODEL_DIR, filename))

@st.cache_data(ttl=300)  # re-check at most every 5 minutes
def check_sample_data_available(url):
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


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

    st.set_page_config(page_title="Telco Churn Classifier", page_icon="📊", layout="wide")
    st.title("📊 Telco Customer Churn - Model Comparison App")
    st.caption("Upload test data, pick a model, and see how it performs.")

    st.sidebar.header("⚙️ Controls")

    if "df" not in st.session_state:
        st.session_state.df = None

    uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)

    sample_available = check_sample_data_available(TEST_DATA_URL)

    if st.sidebar.button("📥 Load sample test data from GitHub", disabled=not sample_available):
        st.session_state.df = pd.read_csv(TEST_DATA_URL)

    if not sample_available:
        st.sidebar.caption("⚠️ Sample data currently unavailable.")

    model_choice = st.sidebar.selectbox("Select a model", list(MODEL_FILES.keys()))

    df = st.session_state.df

    if df is None:
        st.info("Upload a test CSV, or click 'Load sample test data from GitHub' to get started.")
        st.stop()


    st.subheader("Preview of uploaded data")
    st.dataframe(df.head())
    st.divider()
    

    preprocessor = load_preprocessor()
    model = load_model(MODEL_FILES[model_choice])

    try:
        X_proc, y_true = preprocess_input(df, preprocessor)
    except Exception as e:
        st.error(f"Error preprocessing uploaded data: {e}")
        st.stop()

    y_pred = model.predict(X_proc)
   

    st.subheader(f"Predictions:  {model_choice}")

    pred_labels = pd.Series(y_pred).map({0: "No Churn", 1: "Churn"})
    churn_rate = (y_pred == 1).mean()

    if y_true is not None:
        actual_rate = (y_true == 1).mean()
        rate_col1, rate_col2, spacer = st.columns([1, 1, 8])
        rate_col1.metric("Predicted Churn Rate", f"{churn_rate:.1%}")
        rate_col2.metric("Actual Churn Rate", f"{actual_rate:.1%}", delta=f"{(churn_rate - actual_rate):+.1%}", delta_color="off")
    else:
        st.metric("Predicted Churn Rate", f"{churn_rate:.1%}")

    st.caption("Note: closeness between predicted and actual rates doesn't guarantee correct individual predictions, see MCC and the confusion matrix below for actual accuracy.")

    result_preview = df.copy()
    result_preview.insert(0, "Prediction", pred_labels.values)

    if "Churn" in result_preview.columns:
        result_preview.insert(1, "Actual", result_preview["Churn"].map({0: "No Churn", 1: "Churn", "No": "No Churn", "Yes": "Churn"}))
        result_preview.insert(2, "Correct", result_preview["Prediction"] == result_preview["Actual"])

    st.dataframe(result_preview.head(10))

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

    st.sidebar.divider()
    st.sidebar.metric("Rows Processed", len(df))
    st.sidebar.metric("Predicted Churners", int((y_pred == 1).sum()))


if __name__ == "__main__":
    main()