Fraud Detection Using Machine Learning (XGBoost vs Random Forest)
 Project Overview

This project aims to build a Machine Learning–based Fraud Detection System using a real-world dataset from Kaggle.
The dataset contains online transaction records with features such as age, transaction amount, location distance, and transaction hour.

The goal is to predict whether a transaction is fraudulent (1) or not (0).

We trained and compared two ML models:

XGBoost Classifier (with SMOTE oversampling)

Random Forest Classifier (with SMOTE oversampling)

 Dataset Information

The dataset contains the following key columns:

Column Name	Meaning	Example
age	Customer age	32
gender	Male/Female encoded as numbers	0 or 1
merchant	Merchant category	grocery_pos
category	Purchase type	shopping_net
amt	Transaction amount	55.72
city_pop	Population of customer’s city	40,000
trans_hour	Hour of transaction (0–23)	13
dist_disp	Distance between customer and transaction location	2.14 km
fraud	Target label (1 = fraud, 0 = genuine)	1
 Class Imbalance

92% Non-Fraud

8% Fraud

Therefore, we used SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset before training.

 Project Workflow
1️ Import Libraries & Load Dataset

Mounted Google Drive in Colab

Loaded CSV file into DataFrame

Checked shape, data types, null values

2️ Exploratory Data Analysis (EDA)

Performed:

Fraud vs Non-fraud distribution

Feature distributions

Correlations

Boxplots to detect outliers

Found strong imbalance → required SMOTE.

3️ Data Preprocessing

Steps:

Dropped irrelevant columns

Encoded categorical variables

Scaled numerical features (StandardScaler)

Applied SMOTE on training data

4️ Machine Learning Models Trained

Two models were trained:

 Model 1: XGBoost Classifier

Handled with oversampling

Hyperparameters tuned lightly

Used decision-tree boosting

Strengths:
✔ High accuracy
✔ Good recall on fraud class
✔ Handles imbalanced data well

 Model 2: Random Forest Classifier

Ensemble of many decision trees

Good baseline model

Strengths:
✔ Stable and robust
✔ Easy to interpret
✔ Performs well with SMOTE

 Model Evaluation Metrics

We used:

Accuracy

Precision

Recall (MOST important for fraud)

F1 Score

Confusion Matrix

 Expected Results
Metric	XGBoost	Random Forest
Accuracy	Higher	Slightly Lower
Fraud Recall	 Best	Good
Precision	Moderate	Moderate
F1 Score	Best	Good

In fraud detection, Recall is more important:
✔ We must catch as many fraudulent transactions as possible.

 Final Conclusion
Rank	Model	Reason
 1st	XGBoost Classifier	Best recall, best F1 for fraudulent class
 2nd	Random Forest	Good model but slightly weaker recall

Final Model Selected: XGBoost

This model can be deployed for fraud prediction on new transactions.

 How to Run the Project

Follow these steps to run the fraud detection models in Google Colab or locally:

1️ Clone the Repository
git clone https://github.com/your-username/fraud-detection-ml.git
cd fraud-detection-ml

2️ Install Required Libraries

If you are running locally:

pip install -r requirements.txt


If you are running in Google Colab, just run the first code cell in the notebook (it will install everything).

3️ Open the Jupyter/Colab Notebook

Navigate to the notebooks/ folder

Open: fraud_detection_xgboost_rf.ipynb

4️ Upload or Connect Dataset

Upload the CSV file manually OR

Place the dataset inside the data/ folder

Adjust the path in the notebook if needed

5️ Run All Cells

The notebook will automatically:

Preprocess data

Apply SMOTE

Train XGBoost

Train Random Forest

Compare metrics

Display visual results

 Project Structure
fraud-detection-ml/
│
├── data/
│   └── card_transdata.csv          # Raw dataset from Kaggle
│
├── notebooks/
│   └── fraud_detection_xgboost_rf.ipynb   # Full implementation
│
├── models/
│   ├── xgboost_model.pkl           # Saved XGBoost model
│   ├── random_forest_model.pkl     # Saved Random Forest model
│
├── src/
│   ├── preprocess.py               # Preprocessing & SMOTE code (optional)
│   ├── train_xgboost.py            # Script for training XGBoost (optional)
│   ├── train_randomforest.py       # Script for training RF (optional)
│   └── evaluate.py                 # Evaluation functions
│
├── requirements.txt                # Python dependencies
└── README.md                       # Full project documentation (this file)

 Future Improvements

This project can be extended in many ways:

1️ Add More ML Models

LightGBM

CatBoost

Logistic Regression

SVM with class weights

2️ Apply Hyperparameter Tuning

Use:

GridSearchCV

RandomizedSearchCV

Optuna (best and fastest)

3️ Deploy the Model

REST API using Flask / FastAPI

Build a Streamlit dashboard

Deploy to Render / HuggingFace Spaces

4️ Improve Feature Engineering

Time-based features

Behavioral features

Geolocation clustering

5️ Add Explainability Tools (XAI)

SHAP values

LIME

Model interpretability charts

 Bonus: requirements.txt (Copy-Paste)

Add this file to your GitHub:

pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
joblib
