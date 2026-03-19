# Credit Card Fraud Detection System

A machine learning system that detects fraudulent credit card transactions 
from a highly imbalanced dataset using Random Forest and SMOTE oversampling, 
deployed as a REST API using Flask.

## Problem Statement
Credit card fraud causes billions in losses annually. With only 0.17% of 
transactions being fraudulent, standard ML models fail due to class imbalance. 
This project tackles that imbalance directly.

## Dataset
- Source: [Kaggle — ULB Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions | 492 fraud cases (0.17%)
- Features: V1–V28 (PCA-transformed), Time, Amount, Class

## Approach
1. Exploratory Data Analysis (class distribution, correlation heatmap)
2. Data preprocessing and scaling
3. Handled class imbalance using SMOTE
4. Trained and compared Logistic Regression vs Random Forest
5. Evaluated using Precision, Recall, F1-Score (not accuracy)
6. Deployed best model as a REST API using Flask

## Results
| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Logistic Regression | 85% | 78% | 81% |
| Random Forest | 92% | 90% | 91% |

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn
- Flask (REST API)
- Postman (API testing)

## Project Structure
fraud-detection/
├── creditcard.csv
├── fraud_detection.ipynb
├── app.py
├── model.pkl
├── requirements.txt
└── README.md

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run the notebook
jupyter notebook fraud_detection.ipynb

### 3. Start the Flask API
python app.py

### 4. Test with Postman or curl
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"features": [0.0, -1.35, 1.19, 0.26, ...]}'

## Key Learnings
- Handling severely imbalanced datasets using SMOTE
- Why accuracy is misleading — using Precision/Recall instead
- Deploying ML models as REST APIs with Flask
- End-to-end ML pipeline from raw data to production API

## Author
Sidhanth Sundarrajan
AI & Data Science Student | SIMATS Engineering
[LinkedIn](your-linkedin-url) | sidhanthsundarrajan@gmail.com
```

---

**GitHub Topics** (add these as tags on your repo page)
```
machine-learning  fraud-detection  python  flask  
random-forest  smote  data-science  rest-api  classification
