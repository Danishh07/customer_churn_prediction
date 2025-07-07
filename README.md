# Customer Churn Prediction Project 📊

A comprehensive machine learning project that predicts customer churn using the Telco Customer Churn dataset. This project includes data analysis, model training, evaluation, and a Streamlit web application for real-time predictions.

## 🎯 Project Overview

Customer churn prediction helps businesses identify customers who are likely to stop using their services. This project implements multiple machine learning models to predict churn with high accuracy and provides actionable insights for customer retention.

## 🌐 Live Demo

🔗 **Try the deployed app**: [https://customer-churn-system.streamlit.app/](https://customer-churn-system.streamlit.app/)

## ✅ Key Features

- **🤖 Machine Learning Models**: Logistic Regression, Random Forest, and XGBoost
- **📊 Model Performance**: 84.47% ROC-AUC score with comprehensive evaluation metrics
- **🔍 Model Interpretability**: SHAP explanations for transparent decision-making
- **🌐 Interactive Web App**: Streamlit application with reset functionality and improved UI
- **📈 Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **🛠️ Production Ready**: Error handling, input validation, and scalable architecture

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% | 0.8413 |
| Random Forest | 80.06% | 65.87% | 51.60% | 57.87% | 0.8416 |
| **XGBoost (Best)** | **80.20%** | **66.32%** | **51.60%** | **58.05%** | **0.8447** |

## 📁 Project Structure

```
Customer Churn Prediction/
│
├── Customer-Churn.csv          # Dataset file
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── utils.py                    # Helper functions and utilities
├── model.py                    # Model training and evaluation
├── app.py                      # Streamlit web application
│
├── models/                     # Saved trained models
│   ├── best_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── model_metadata.pkl
│
└── images/                     # Generated visualizations
    ├── eda_plots.png
    ├── categorical_analysis.png
    ├── correlation_heatmap.png
    ├── roc_curves.png
    ├── shap_summary_plot.png
    ├── shap_bar_plot.png
    └── *_feature_importance.png
```

## 🛠️ Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Clone or Download the Project
Download all project files to your local directory.

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Dataset
Ensure `Customer-Churn.csv` is in the project root directory.

## 🎮 Usage

### 1. Train the Models
Run the model training script to train all models and generate analysis:

```bash
python model.py
```

This will:
- Load and explore the dataset
- Perform comprehensive EDA
- Train Logistic Regression, Random Forest, and XGBoost models
- Evaluate models with multiple metrics
- Generate SHAP explanations
- Save trained models and visualizations

### 2. Launch the Web Application
Start the Streamlit app for interactive predictions:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Make Predictions
Use the web app to:
- Enter customer information manually
- Select from sample data
- Get churn predictions with probability scores
- View model explanations and recommendations

## 📊 Model Performance

The project trains three models and selects the best performer:

- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: Gradient boosting with high performance

Models are evaluated using:
- Accuracy
- Precision, Recall, F1-score
- ROC-AUC score
- Confusion Matrix

## 🔍 Features Used

The model uses 19+ customer features including:

**Demographics:**
- Gender, Senior Citizen status
- Partner and Dependents information

**Account Information:**
- Tenure (months with company)
- Contract type (Month-to-month, One year, Two year)
- Payment method
- Paperless billing preference

**Services:**
- Phone service and multiple lines
- Internet service type
- Add-on services (Online security, backup, etc.)
- Streaming services

**Financial:**
- Monthly charges
- Total charges

## 🎨 Web App Features

The Streamlit app includes:

### 🔮 Prediction Page
- Interactive input form for customer data
- Sample data selection
- Real-time churn prediction
- Probability visualization
- Customer profile summary
- Actionable recommendations

### 📊 Model Insights
- Model performance metrics
- Feature importance analysis
- SHAP explanation plots

### ℹ️ About Page
- Project description
- Technical details
- Usage instructions

## 📈 Key Insights

From the analysis, key churn factors include:
- **Contract Type**: Month-to-month contracts have higher churn
- **Tenure**: Newer customers are more likely to churn
- **Payment Method**: Electronic check users churn more
- **Monthly Charges**: Higher charges increase churn probability
- **Internet Service**: Fiber optic users show different patterns

## 🎯 Business Impact

This model helps businesses:
- **Identify At-Risk Customers**: Proactive retention strategies
- **Optimize Resources**: Focus efforts on high-risk customers
- **Improve Customer Experience**: Address pain points causing churn
- **Increase Revenue**: Reduce customer acquisition costs

## 🛡️ Model Interpretability

The project uses SHAP (SHapley Additive exPlanations) to provide:
- Feature importance rankings
- Individual prediction explanations
- Global model behavior understanding
- Transparent decision-making process

## 🚀 Deployment Ready

The Streamlit app is ready for deployment on platforms like:
- Streamlit Cloud
- Heroku
- AWS
- Google Cloud Platform
- Azure


## 🎯 Key Business Insights

### Top Churn Risk Factors:
1. **Contract Type**: Month-to-month contracts (highest risk)
2. **Tenure**: New customers (< 12 months)
3. **Payment Method**: Electronic check users
4. **Monthly Charges**: Higher charges increase churn probability
5. **Internet Service**: Fiber optic customers show different patterns

### Actionable Recommendations:
- **High-Risk Customers**: Immediate retention calls, special offers
- **New Customers**: Enhanced onboarding and support
- **Contract Optimization**: Incentivize longer-term contracts
- **Payment Methods**: Encourage automatic payment methods
- **Service Quality**: Focus on customer satisfaction metrics

## 🙏 Acknowledgments

- Telco Customer Churn Dataset from Kaggle
- scikit-learn for machine learning algorithms
- SHAP for model interpretability
- Streamlit for the web application framework
- Plotly for interactive visualizations

---

**🎯 Mission Accomplished!** This project provides a comprehensive solution for customer churn prediction with a beautiful, functional web interface ready for deployment.
- **Automated Retraining**: Regular model updates
- **API Integration**: RESTful API for external systems

---

**Happy Predicting! 🎉**
