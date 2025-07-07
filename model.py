"""
Model training and evaluation for Customer Churn Prediction.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import shap
import matplotlib.pyplot as plt
from utils import load_and_explore_data, clean_data, perform_eda, preprocess_features, evaluate_model, plot_roc_curves, create_feature_importance_plot
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    """
    Customer Churn Prediction Model Class
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.preprocessors = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_and_prepare_data(self, file_path):
        """
        Load and prepare data for training.
        
        Args:
            file_path (str): Path to the dataset CSV file
        """
        try:
            print("Loading and exploring data...")
            df = load_and_explore_data(file_path)
            
            if df is None:
                raise Exception("Failed to load data")
            
            print("\nCleaning data...")
            df_clean = clean_data(df)
            
            print("\nPerforming EDA...")
            perform_eda(df_clean)
            
            print("\nPreprocessing features...")
            X, y, feature_names, preprocessors = preprocess_features(df_clean)
            
            if X is None:
                raise Exception("Failed to preprocess features")
            
            self.feature_names = feature_names
            self.preprocessors = preprocessors
            
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            print(f"Training set shape: {self.X_train.shape}")
            print(f"Test set shape: {self.X_test.shape}")
            print(f"Training set churn rate: {self.y_train.mean():.3f}")
            print(f"Test set churn rate: {self.y_test.mean():.3f}")
            
        except Exception as e:
            print(f"Error in data preparation: {e}")
            raise
    
    def train_models(self):
        """
        Train multiple models for churn prediction.
        """
        try:
            print("\nTraining models...")
            
            # Define models with hyperparameters
            model_configs = {
                'Logistic Regression': {
                    'model': LogisticRegression(random_state=42, max_iter=1000),
                    'params': {
                        'C': [0.1, 1, 10],
                        'solver': ['liblinear', 'lbfgs']
                    }
                },
                'Random Forest': {
                    'model': RandomForestClassifier(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5]
                    }
                },
                'XGBoost': {
                    'model': XGBClassifier(random_state=42, eval_metric='logloss'),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 6, 10],
                        'learning_rate': [0.01, 0.1, 0.2]
                    }
                }
            }
            
            results = {}
            
            for model_name, config in model_configs.items():
                print(f"\nTraining {model_name}...")
                
                # Perform GridSearchCV for hyperparameter tuning
                grid_search = GridSearchCV(
                    config['model'], 
                    config['params'], 
                    cv=5, 
                    scoring='roc_auc',
                    n_jobs=-1
                )
                
                grid_search.fit(self.X_train, self.y_train)
                
                # Store the best model
                self.models[model_name] = grid_search.best_estimator_
                
                print(f"Best parameters for {model_name}: {grid_search.best_params_}")
                print(f"Best CV score for {model_name}: {grid_search.best_score_:.4f}")
                
                # Evaluate the model
                model_results = evaluate_model(
                    self.models[model_name], 
                    self.X_test, 
                    self.y_test, 
                    model_name
                )
                
                results[model_name] = model_results
                
                # Generate detailed classification report
                y_pred = self.models[model_name].predict(self.X_test)
                print(f"\nDetailed Classification Report for {model_name}:")
                print(classification_report(self.y_test, y_pred))
            
            # Plot ROC curves
            plot_roc_curves(results, self.y_test)
            
            # Find the best model based on ROC-AUC score
            best_roc_auc = 0
            for model_name, result in results.items():
                if result and result['roc_auc'] and result['roc_auc'] > best_roc_auc:
                    best_roc_auc = result['roc_auc']
                    self.best_model = self.models[model_name]
                    self.best_model_name = model_name
            
            print(f"\nBest model: {self.best_model_name} with ROC-AUC: {best_roc_auc:.4f}")
            
            # Create feature importance plots for tree-based models
            for model_name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    create_feature_importance_plot(model, self.feature_names, model_name)
            
            return results
            
        except Exception as e:
            print(f"Error during model training: {e}")
            raise
    
    def explain_predictions_with_shap(self, sample_size=100):
        """
        Use SHAP to explain XGBoost predictions.
        
        Args:
            sample_size (int): Number of samples to use for SHAP explanation
        """
        try:
            if 'XGBoost' not in self.models:
                print("XGBoost model not found. SHAP explanation requires XGBoost.")
                return None
            
            print(f"\nGenerating SHAP explanations for XGBoost...")
            
            # Ensure images directory exists
            import os
            if not os.path.exists('images'):
                os.makedirs('images')
            
            # Use a subset of test data for SHAP (for performance)
            X_sample = self.X_test.sample(n=min(sample_size, len(self.X_test)), random_state=42)
            
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.models['XGBoost'])
            shap_values = explainer.shap_values(X_sample)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
            plt.title('SHAP Summary Plot - Feature Importance for Churn Prediction')
            plt.tight_layout()
            plt.savefig('images/shap_summary_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Bar plot for a single prediction
            plt.figure(figsize=(10, 8))
            shap.plots.bar(shap_values[0], feature_names=self.feature_names, show=False)
            plt.title('SHAP Bar Plot - Single Prediction Explanation')
            plt.tight_layout()
            plt.savefig('images/shap_bar_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("SHAP plots saved successfully in images/ directory!")
            
            return explainer, shap_values
            
        except Exception as e:
            print(f"Error generating SHAP explanations: {e}")
            return None, None
    
    def save_models(self, save_path='models/'):
        """
        Save trained models and preprocessors.
        
        Args:
            save_path (str): Directory to save models
        """
        try:
            import os
            os.makedirs(save_path, exist_ok=True)
            
            # Save all models
            for model_name, model in self.models.items():
                model_filename = f"{save_path}{model_name.lower().replace(' ', '_')}_model.pkl"
                joblib.dump(model, model_filename)
                print(f"Saved {model_name} to {model_filename}")
            
            # Save the best model separately
            if self.best_model:
                best_model_filename = f"{save_path}best_model.pkl"
                joblib.dump(self.best_model, best_model_filename)
                print(f"Saved best model ({self.best_model_name}) to {best_model_filename}")
            
            # Save feature names and preprocessors
            metadata = {
                'feature_names': self.feature_names,
                'preprocessors': self.preprocessors,
                'best_model_name': self.best_model_name
            }
            metadata_filename = f"{save_path}model_metadata.pkl"
            joblib.dump(metadata, metadata_filename)
            print(f"Saved metadata to {metadata_filename}")
            
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def load_models(self, save_path='models/'):
        """
        Load trained models and preprocessors.
        
        Args:
            save_path (str): Directory to load models from
        """
        try:
            # Load metadata
            metadata_filename = f"{save_path}model_metadata.pkl"
            metadata = joblib.load(metadata_filename)
            
            self.feature_names = metadata['feature_names']
            self.preprocessors = metadata['preprocessors']
            self.best_model_name = metadata['best_model_name']
            
            # Load best model
            best_model_filename = f"{save_path}best_model.pkl"
            self.best_model = joblib.load(best_model_filename)
            
            print(f"Loaded best model: {self.best_model_name}")
            print(f"Feature names loaded: {len(self.feature_names)} features")
            
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def predict_churn(self, customer_data):
        """
        Predict churn for a single customer.
        
        Args:
            customer_data (dict): Dictionary containing customer features
            
        Returns:
            tuple: (prediction, probability)
        """
        try:
            if self.best_model is None:
                raise Exception("No model loaded. Please train or load a model first.")
            
            # Convert customer data to DataFrame
            customer_df = pd.DataFrame([customer_data])
            
            # Convert binary categorical variables to numeric
            binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
            for col in binary_cols:
                if col in customer_df.columns:
                    customer_df[col] = customer_df[col].map({'Yes': 1, 'No': 0})
            
            # Convert gender to numeric
            if 'gender' in customer_df.columns:
                customer_df['gender'] = customer_df['gender'].map({'Male': 1, 'Female': 0})
            
            # Apply the same preprocessing as training data
            # Handle categorical encoding
            categorical_cols = self.preprocessors['categorical_cols']
            customer_encoded = pd.get_dummies(customer_df, columns=categorical_cols, drop_first=True)
            
            # Ensure all features are present (add missing columns with 0)
            for feature in self.feature_names:
                if feature not in customer_encoded.columns:
                    customer_encoded[feature] = 0
            
            # Reorder columns to match training data
            customer_encoded = customer_encoded[self.feature_names]
            
            # Scale numerical features (only the ones that were scaled during training)
            numerical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
                            'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges']
            
            # Only scale the numerical columns that exist in the feature set
            numerical_cols_to_scale = [col for col in numerical_cols if col in customer_encoded.columns]
            
            if numerical_cols_to_scale and 'scaler' in self.preprocessors:
                customer_encoded[numerical_cols_to_scale] = self.preprocessors['scaler'].transform(
                    customer_encoded[numerical_cols_to_scale]
                )
            
            # Make prediction
            prediction = self.best_model.predict(customer_encoded)[0]
            probability = self.best_model.predict_proba(customer_encoded)[0]
            
            return prediction, probability
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, None

def main():
    """
    Main function to run the complete pipeline.
    """
    try:
        # Initialize the predictor
        predictor = ChurnPredictor()
        
        # Load and prepare data
        predictor.load_and_prepare_data('Customer-Churn.csv')
        
        # Train models
        results = predictor.train_models()
        
        # Generate SHAP explanations
        predictor.explain_predictions_with_shap()
        
        # Save models
        predictor.save_models()
        
        print("\n" + "="*50)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Best model: {predictor.best_model_name}")
        print("Models saved to 'models/' directory")
        print("Visualization plots saved as PNG files")
        print("Ready to run the Streamlit app!")
        
    except Exception as e:
        print(f"Error in main pipeline: {e}")

if __name__ == "__main__":
    main()
