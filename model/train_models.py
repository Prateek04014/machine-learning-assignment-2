#base load
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

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

#config load
DATA_PATH = "telco_churn_raw.csv"   #csv location
RANDOM_STATE = 42   #random seed
TEST_SIZE = 0.2     #20% data for testing

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

#function definition
def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])    #Dropping as it is just an identifier and not a feature
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce") #Converted to numbers, was read as object due to blanks.
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
   #df["TotalCharges"] = df["TotalCharges"].fillna(0)   #Customers with 0 tenure have not accumulated any charges yet
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})  #Convert target to 1/0

    return df

def setup_preprocessor(df, target_col="Churn"):
    feature_df = df.drop(columns=[target_col])

    numeric_cols = feature_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols), #handle_unknown="ignore" handles crash of streamlit
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


def main():
    print("Getting the data ready...")
    df = load_and_clean(os.path.join(OUTPUT_DIR, DATA_PATH))

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )   

    raw_test = X_test.copy()
    raw_test["Churn"] = y_test.values
    test_csv_path = os.path.join(os.path.dirname(OUTPUT_DIR), "test_data.csv")
    raw_test.to_csv(test_csv_path, index=False)
    print(f"Saved raw test data to {test_csv_path}")

    preprocessor, numeric_cols, categorical_cols = setup_preprocessor(df)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)


   #test
    print(f"Shape: {df.shape}")
    print(f"Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Churn value counts:\n{df['Churn'].value_counts()}")
    print(f"TotalCharges dtype: {df['TotalCharges'].dtype}")
    print(f"Train shape: {X_train.shape}   Test shape: {X_test.shape}")
    print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
    print(f"After encoding — X_train: {X_train_proc.shape}   X_test: {X_test_proc.shape}")

if __name__ == "__main__":
    main()