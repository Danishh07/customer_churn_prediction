"""
Utility functions for the Customer Churn Prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data(file_path):
    """
    Load the dataset and perform initial exploration.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)
        
        print("Dataset Shape:", df.shape)
        print("\nDataset Info:")
        print(df.info())
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nMissing values:")
        print(df.isnull().sum())
        print("\nChurn distribution:")
        print(df['Churn'].value_counts())
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """
    Clean and preprocess the dataset.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    try:
        # Make a copy to avoid modifying original data
        df_clean = df.copy()
        
        # Convert TotalCharges to numeric (it might be stored as string)
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
        
        # Handle missing values in TotalCharges
        if df_clean['TotalCharges'].isnull().sum() > 0:
            # Fill missing TotalCharges with median
            median_charges = df_clean['TotalCharges'].median()
            df_clean['TotalCharges'].fillna(median_charges, inplace=True)
            print(f"Filled {df_clean['TotalCharges'].isnull().sum()} missing values in TotalCharges with median: {median_charges}")
        
        # Convert binary categorical variables to numeric
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
        for col in binary_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})
        
        # Convert gender to numeric
        if 'gender' in df_clean.columns:
            df_clean['gender'] = df_clean['gender'].map({'Male': 1, 'Female': 0})
        
        print("Data cleaning completed successfully!")
        return df_clean
    
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return df

def perform_eda(df):
    """
    Perform Exploratory Data Analysis and create visualizations.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
    """
    try:
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Churn Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Churn distribution
        churn_counts = df['Churn'].value_counts()
        axes[0,0].pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Churn Distribution')
        
        # Tenure distribution by churn
        sns.histplot(data=df, x='tenure', hue='Churn', bins=30, alpha=0.7, ax=axes[0,1])
        axes[0,1].set_title('Tenure Distribution by Churn')
        
        # Monthly charges distribution by churn
        sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[1,0])
        axes[1,0].set_title('Monthly Charges by Churn')
        
        # Total charges distribution by churn
        sns.boxplot(data=df, x='Churn', y='TotalCharges', ax=axes[1,1])
        axes[1,1].set_title('Total Charges by Churn')
        
        plt.tight_layout()
        plt.savefig('eda_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Categorical variables analysis
        categorical_cols = ['Contract', 'PaymentMethod', 'InternetService']
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, col in enumerate(categorical_cols):
            if col in df.columns:
                churn_by_cat = df.groupby([col, 'Churn']).size().unstack().fillna(0)
                churn_by_cat.plot(kind='bar', ax=axes[i], rot=45)
                axes[i].set_title(f'Churn by {col}')
                axes[i].legend(['No Churn', 'Churn'])
        
        plt.tight_layout()
        plt.savefig('categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Correlation heatmap for numerical features
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
        if all(col in df.columns for col in numerical_cols):
            plt.figure(figsize=(10, 8))
            correlation_matrix = df[numerical_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, linewidths=0.5)
            plt.title('Correlation Heatmap of Numerical Features')
            plt.tight_layout()
            plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("EDA completed successfully! Plots saved as PNG files.")
        
    except Exception as e:
        print(f"Error during EDA: {e}")

def preprocess_features(df, target_column='Churn'):
    """
    Preprocess features for machine learning.
    
    Args:
        df (pd.DataFrame): Cleaned dataset
        target_column (str): Name of target column
        
    Returns:
        tuple: (X_processed, y, feature_names, preprocessors)
    """
    try:
        # Separate features and target
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Identify categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        print(f"Categorical columns: {categorical_cols}")
        print(f"Numerical columns: {numerical_cols}")
        
        # One-hot encode categorical variables
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Scale numerical features
        scaler = StandardScaler()
        if numerical_cols:
            X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
        
        print(f"Features after preprocessing: {X_encoded.shape}")
        print(f"Feature names: {list(X_encoded.columns)}")
        
        return X_encoded, y, list(X_encoded.columns), {'scaler': scaler, 'categorical_cols': categorical_cols}
    
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None, None, None, None

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate a trained model and return metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_name (str): Name of the model for printing
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # ROC-AUC score
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n{model_name} Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        if roc_auc:
            print(f"ROC-AUC: {roc_auc:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
    
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return None

def plot_roc_curves(models_results, y_test):
    """
    Plot ROC curves for multiple models.
    
    Args:
        models_results (dict): Dictionary containing model results
        y_test: True test labels
    """
    try:
        plt.figure(figsize=(10, 8))
        
        for model_name, results in models_results.items():
            if results and results['prediction_probabilities'] is not None:
                fpr, tpr, _ = roc_curve(y_test, results['prediction_probabilities'])
                roc_auc = results['roc_auc']
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("ROC curves plot saved as 'roc_curves.png'")
        
    except Exception as e:
        print(f"Error plotting ROC curves: {e}")

def create_feature_importance_plot(model, feature_names, model_name="Model"):
    """
    Create feature importance plot for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        model_name (str): Name of the model
    """
    try:
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, y='feature', x='importance')
            plt.title(f'Top 20 Feature Importances - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(f'{model_name.lower()}_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Feature importance plot saved for {model_name}")
            
        else:
            print(f"Model {model_name} does not have feature_importances_ attribute")
    
    except Exception as e:
        print(f"Error creating feature importance plot: {e}")
