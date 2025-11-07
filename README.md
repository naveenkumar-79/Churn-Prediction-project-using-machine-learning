# Churn-Prediction-project-using-machine-learning
🏗️ Project Architecture
Customer Churn Prediction
│
├── app.py                     # Flask application file
├── Churn Prediction.pkl       # Trained machine learning model (Logistic Regression)
├── standard_scalar.pkl        # Fitted StandardScaler object
├── templates/
│   ├── index.html             # Web interface for user input
│   └── result.html            # Prediction output page
├── static/
│   └── style.css              # Styling for the web app
├── requirements.txt           # Python dependencies
├── Procfile                   # Command for deployment (Gunicorn)
├── README.md                  # Project documentation
└── dataset/
    └── churn.csv              # Source data (Telco Customer dataset)

✨ Key Features

End-to-end ML pipeline from raw data to deployment

Data visualization using Matplotlib and Seaborn

Robust preprocessing:

Missing value handling (Iterative Imputer)

Outlier treatment (Winsorizer – Gaussian capping)

Variable transformation (Quantile Transformer)

Data balancing with SMOTE

Feature scaling using StandardScaler

Feature selection using Chi-Square Hypothesis Testing

Model tuning via GridSearchCV

Deployment-ready with Flask + Gunicorn

⚙️ Requirements
Python Packages Used
blinker==1.9.0
click==8.3.0
colorama==0.4.6
feature_engine==1.9.3
Flask==3.1.2
gunicorn==23.0.0
imbalanced-learn==0.14.0
itsdangerous==2.2.0
Jinja2==3.1.4
joblib==1.4.2
matplotlib==3.9.2
numpy==2.1.3
pandas==2.2.3
scikit-learn==1.5.2
scipy==1.14.1
seaborn==0.13.2
statsmodels==0.14.3
xgboost==2.1.3


Install dependencies with:

pip install -r requirements.txt

📊 Data Preparation Steps

Data Cleaning

Removed noise and irrelevant columns

Handled null values using Iterative Imputer (DecisionTreeRegressor)

Feature Engineering

Separated categorical and numerical data

Applied Quantile Transformation for normalization

Encoded categorical data:

OneHotEncoder → Nominal features

OrdinalEncoder → Ordered features

LabelEncoder → Target variable

Outlier Handling

Used Winsorizer (Gaussian capping) to control extreme values

Feature Selection

Removed less significant variables using Chi-Square test (p < 0.05)

Data Balancing

Balanced imbalanced classes using SMOTE (Synthetic Minority Oversampling Technique)

Feature Scaling

Standardized numerical features with StandardScaler

🧩 Model Development
Step	Description
Model Type	Binary Classification (Churn vs Non-Churn)
Algorithms Tested	Logistic Regression, Random Forest, Decision Tree, Naïve Bayes, XGBoost
Evaluation Metrics	Accuracy, ROC Curve, AUC Score
Final Model	Logistic Regression (AUC = 0.75)
Model Saving	Pickled using pickle.dump() for Flask deployment
🎯 Model Tuning

Used GridSearchCV to optimize Logistic Regression parameters:

param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'solver': ['liblinear', 'lbfgs', 'newton-cg', 'saga']
}

🌐 Web Deployment
Flask Application (app.py)

Takes user inputs through HTML form

Scales features using the saved standard_scalar.pkl

Predicts churn with the trained Logistic Regression model (Churn Prediction.pkl)

Displays prediction result dynamically

Procfile (for Heroku / Render Deployment)
web: gunicorn app:app

Run Locally
python app.py


Then open your browser and navigate to:

http://127.0.0.1:5000/

🧱 Model Workflow
graph TD
A[Data Collection] --> B[Data Cleaning & Imputation]
B --> C[Feature Engineering & Encoding]
C --> D[Outlier & Scaling]
D --> E[Feature Selection & Hypothesis Testing]
E --> F[Model Training & Evaluation]
F --> G[Model Tuning (GridSearchCV)]
G --> H[Model Deployment with Flask]

📈 Results

Best Model: Logistic Regression

Accuracy: 75%

ROC-AUC: 0.75

Key Insight:
Customers with month-to-month contracts, higher monthly charges, and paperless billing are most likely to churn.

🧪 Future Enhancements

Integrate Deep Learning models for better churn prediction

Add real-time dashboard using Streamlit or Dash

Deploy on Docker and Kubernetes for scalability

Integrate with CRM systems for automated customer retention campaigns

👨‍💻 Author

P. Naveen Kumar
Under the guidance of Vihara Tech Institute
📧 [puppalanaveenkumar11@gmail.com]
