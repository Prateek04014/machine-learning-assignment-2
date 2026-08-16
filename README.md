# Telco Customer Churn — Classification Model Comparison

## a. Problem Statement

Every year telecom companies lose a lot of customers to churn (rate at which customers stop using a telecommunications company's services (such as mobile, broadband, landline, or TV) over a given period).

Being able to predict which customers are likely to leave allows a company to proactively intervene (offers, retention calls, etc.). This project frames churn prediction as a binary classification problem and compares several ML models to determine which is most effective on this dataset.

## b. Dataset Description

- **Source:** [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn?select=WA_Fn-UseC_-Telco-Customer-Churn.csv) (Kaggle)
- **Instances:** 7,043 customers
- **Features:** 19 (after dropping the `customerID` identifier column)
- **Target:** `Churn` (Yes/No — encoded to 1/0)
- **Feature types:** Mix of numeric (`tenure`, `MonthlyCharges`, `TotalCharges`) and
  categorical (`gender`, `Contract`, `InternetService`, `PaymentMethod`, etc.)
- **Class balance:** ~26.5% churn, ~73.5% no-churn (moderately imbalanced)
- **Preprocessing steps applied:**
  - Dropped `customerID` (not a predictive feature)
  - Converted `TotalCharges` from string to numeric; imputed missing values
    (blank for customers with 0 tenure) with the median
  - Standard-scaled numeric features
  - One-hot encoded categorical features

The Telco dataset had a good blend of categorical and numeric features with a good enough size (7043 rows). This provided a close to real life scenario while comfortably meeting minimum size and feature requirements.

## c. GitHub Repository Link
- [Repo Link](https://github.com/Prateek04014/machine-learning-assignment-2)
## Project Structure

```
machine-learning-assignment-2/
│-- app.py                     # Streamlit app
│-- requirements.txt
│-- README.md
│-- test_data.csv              # test data split (raw, unprocessed)
│-- .gitignore
│-- model/
    │-- train_models.py        # Loads data, trains all 5 models, saves artifacts
    │-- telco_churn_raw.csv    # Full raw dataset
    │-- preprocessor.pkl       # pickle files used to store trained models for app.py to use
    │-- logistic_regression.pkl
    │-- decision_tree.pkl
    │-- knn.pkl
    │-- naive_bayes.pkl
    │-- random_forest.pkl
    │-- metrics_summary.csv    # Comparison table output
```

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8055 | 0.8419 | 0.6572 | 0.5588 | 0.6040 | 0.4790 |
| Decision Tree | 0.7289 | 0.6573 | 0.4896 | 0.5053 | 0.4974 | 0.3119 |
| kNN | 0.7637 | 0.7896 | 0.5527 | 0.5749 | 0.5636 | 0.4018 |
| Naive Bayes | 0.6948 | 0.8074 | 0.4589 | 0.8369 | 0.5928 | 0.4245 |
| Random Forest (Ensemble) | 0.7750 | 0.8187 | 0.5979 | 0.4652 | 0.5233 | 0.3842 |

This data also gets stored in `model/metrics_summary.csv` after running `model/train_models.py`.
The training data and the test data are split by an 80-20 ratio with a fixed seed 42 (to ensure repeatability)

### Observations on Model Performance

| ML Model Name | Observation |
|---|---|
| Logistic Regression | It has the highest across every metric except Recall. Since many features present have a linear relation with churn (target variable), it seems to show such performance. |
| Decision Tree | Has the weakest AUC & MCC. Likely cause is it being a single decision tree, without pruning, tends to overfit the training data|
| kNN | kNN is in the middle of the pack. Since kNN is a distance based algorithm, standardisation was important. It classifies a customer based on the majority class among its k nearest neighbors in the feature space. Its effectiveness can decrease overtime as the dataset grows more complex. kNN's biggest weakness is high dimensionality, making distance less meaningful and computationally expensive|
| Naive Bayes | Naive Bayes got the highest Recall but the lowest precision, this signifies it prioritises minimising missed churns over minimising false alarms |
| Random Forest (Ensemble) | Random Forest improves upon the decision tree by building many different decision trees and combining their predictions. Since each tree is different, errors tend to cancel out, leading to a more stable outcome that is less prone to overfitting. The biggest tradeoff for this is the lowest recall score it got (0.4652) |
| **Overall Winner for the dataset?** | Logistic Regression seems to best fit for this dataset. Based on all the metrics, it has the highest accuracy (0.8055), the highest AUC (0.8419), Precision (0.6572) and MCC (0.4790). That said, if the business prioritises catching as many at-risk customers as possible, even at the cost of false alarms, Naive Bayes much higher Recall score (0.8369) could make it a more practical choice. |

Looking at all the above models, there seems to be a clear precision-recall tradeoff pattern emerging. Naive Bayes has the highest recall but sacrifices precision, while Random Forest is the opposite, trading stability for recall. Logistic Regression is the only model that performs well across both, which is why it wins overall.

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Prateek04014/machine-learning-assignment-2.git
cd machine-learning-assignment-2

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train models (generates .pkl files, preprocessor, and metrics_summary.csv)
python model/train_models.py

# Run the Streamlit app
streamlit run app.py
```

## Live App

[Live Streamlit App](https://machine-learning-assignment-2-telco-churn.streamlit.app/)

"Upload a test CSV, or use the sample-data button in the sidebar to get started immediately."


## App Features

- **One-click sample data loading** directly from the repo. A check is also built-in, greying out the button if the data is not available
- **Predicted vs. Actual churn rate comparison** shown to compare the models with more ease
- **Row-level Prediction / Actual / Correct columns** for row level checks for each model
- **Graceful error handling**: Wrong dataset or dataset with missing columns is handled gracefully. Extra columns are ignored
- **Classification report table** alongside the confusion matrix to show a better breakdown