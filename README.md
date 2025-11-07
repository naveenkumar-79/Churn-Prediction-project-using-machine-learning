# 📊 Customer Churn Prediction Using Machine Learning

> A Machine Learning project to predict telecom customer churn, enabling businesses to take proactive actions for customer retention.

---

## 🧠 Project Overview

Customer churn is a major challenge for subscription-based businesses. This project leverages *Machine Learning* techniques to identify customers likely to discontinue services. By analyzing demographic, behavioral, and billing data, the model provides insights that help businesses *enhance retention strategies, **personalize offers, and **reduce churn rates*.

🚀 *Goal:* Predict whether a customer will churn (Yes/No)
📈 *Final Model:* Logistic Regression
🎯 *Accuracy:* ~75%

---

## 🧩 Architecture

mermaid
graph TD
A[Data Collection] --> B[Data Cleaning & Imputation]
B --> C[Feature Engineering]
C --> D[Feature Selection]
D --> E[Data Balancing (SMOTE)]
E --> F[Feature Scaling (StandardScaler)]
F --> G[Model Training - Logistic Regression]
G --> H[Model Evaluation (AUC/ROC)]
H --> I[Model Deployment - Flask + Render]


---

## 🧰 Tech Stack & Libraries

| Category                | Tools / Libraries                          |
| ----------------------- | ------------------------------------------ |
| *Language*            | Python 3                                   |
| *Framework*           | Flask                                      |
| *ML Libraries*        | Scikit-learn, XGBoost, Statsmodels         |
| *EDA & Visualization* | Matplotlib, Seaborn, Pandas                |
| *Feature Engineering* | Feature-Engine, Imbalanced-learn           |
| *Deployment*          | Render, Gunicorn                           |
| *Others*              | Joblib, Pickle, Numpy, SciPy, Flask-Jinja2 |

---

## 🗃 Dataset Description

* *Source:* Telco Customer Churn dataset (public dataset)
* *Records:* 7043 rows × 21 columns
* *Target Variable:* Churn
* *Features:*

  * Demographics: Gender, SeniorCitizen, Dependents, Partner
  * Services: InternetService, OnlineSecurity, TechSupport, etc.
  * Contract details: Contract Type, Payment Method
  * Charges: MonthlyCharges, TotalCharges

---

## 📊 Data Preprocessing Steps

1. *Missing Value Handling:* Iterative Imputer (DecisionTreeRegressor)
2. *Outlier Treatment:* Winsorizer (Gaussian, fold=2.5)
3. *Feature Transformation:* Quantile Transformer (Normal Distribution)
4. *Encoding:*

   * OneHotEncoding → Nominal variables
   * OrdinalEncoding → Contract type
   * LabelEncoding → Target variable
5. *Feature Selection:* Chi-Square Test, Correlation Analysis
6. *Data Balancing:* SMOTE
7. *Scaling:* StandardScaler (Z-Score normalization)

---

## 🧮 Model Training

* Trained multiple algorithms:

  * KNN Classifier
  * GaussianNB
  * Logistic Regression ✅ (Best Model)
  * Decision Tree
  * Random Forest
  * Gradient Boosting
  * XGBoost
  * SVC

* *Evaluation Metric:* AUC-ROC Curve

* *Best Performing Model:* Logistic Regression

* *Saved Model:* model.pkl

* *Scaler:* scaler.pkl

---

## 🧾 Model Evaluation

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 75%   |
| Precision | 0.74  |
| Recall    | 0.76  |
| AUC Score | 0.82  |

🧭 *ROC Curve & AUC Visualization:*
![ROC Curve Placeholder](https://placehold.co/600x300?text=ROC+Curve+Placeholder)

---

## 💻 Flask Deployment

This model is deployed using *Flask* (backend) and *Render* (cloud platform).

*Deployment Steps:*

1. Build Flask app (app.py)
2. Create HTML templates (index.html, result.html, etc.)
3. Push project to GitHub
4. Deploy on Render:

   bash
   gunicorn app:app
   

🌐 *Live Demo:* [https://your-render-app-url.onrender.com](#) (replace with your link)

---

## 🧩 Project Structure


├── app.py
├── templates/
│   ├── index.html
│   ├── result.html
│   ├── about.html
├── static/
│   ├── style.css
│   ├── visualizations/
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── README.md
└── churn_data.csv


---

## 📈 Visual Insights

Below are some insights derived during Exploratory Data Analysis (EDA):

* 🔹 *Senior citizens* and *female customers* have higher churn rates.
* 🔹 Customers with *month-to-month contracts* are more likely to churn.
* 🔹 *Higher monthly charges* correlate with higher churn probability.
* 🔹 Customers with *multiple phone lines* or *fiber optic internet* show higher churn.
* 🔹 *Longer tenure* customers are more loyal.

📊 *Sample Visualization:*
![EDA Dashboard Placeholder](https://placehold.co/700x350?text=EDA+Visualization+Placeholder)

---

## ⚙ Installation & Usage

bash
# Clone the repository
git clone https://github.com/username/customer-churn-prediction.git
cd customer-churn-prediction

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py


Then open: http://127.0.0.1:5000/

---

## 🧑‍💻 Author

*👨‍🎓 P. Naveen Kumar*
Under the guidance of *Vihara Tech Institute*
📍 A dissertation submitted to The Skill Union
💬 "From reactive to proactive retention strategies through data-driven intelligence."

---

## 📜 License

This project is licensed under the *MIT License* — feel free to use and modify with credit.

---

## 🌟 Acknowledgements

* *Vihara Tech Institute* for project guidance
* *Scikit-learn* and *Pandas* communities
* *Render* for deployment infrastructure
* *Matplotlib & Seaborn* for visualization power

---

## 🧭 Future Enhancements

* Add SHAP & LIME explainability dashboards
* Build an interactive churn analytics dashboard using Plotly/Dash
* Integrate auto-retraining pipeline with new data
* Deploy using Docker + CI/CD

---

⭐ *If you like this project, give it a star on GitHub!*
