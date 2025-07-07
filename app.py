"""
Streamlit app for Customer Churn Prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from model import ChurnPredictor
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .churn-high {
        background: #fff5f5;
        border-left: 5px solid #dc2626;
        color: #7f1d1d;
    }
    .churn-high h3 {
        color: #dc2626 !important;
    }
    .churn-low {
        background: #f0fdf4;
        border-left: 5px solid #16a34a;
        color: #14532d;
    }
    .churn-low h3 {
        color: #16a34a !important;
    }
    .reset-button {
        background-color: #6b7280;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
    }
    .reset-button:hover {
        background-color: #4b5563;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_model_and_metadata():
    """Load the trained model and metadata."""
    try:
        predictor = ChurnPredictor()
        predictor.load_models('models/')
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_sample_data():
    """Load sample data for demonstration."""
    try:
        df = pd.read_csv('Customer-Churn.csv')
        return df.head(5)
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None

def create_feature_explanation():
    """Create a feature explanation section."""
    with st.expander("📝 Feature Explanations"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Customer Demographics:**
            - **Gender**: Customer's gender (Male/Female)
            - **Senior Citizen**: Whether customer is 65 or older (0=No, 1=Yes)
            - **Partner**: Whether customer has a partner (Yes/No)
            - **Dependents**: Whether customer has dependents (Yes/No)
            
            **Services:**
            - **Phone Service**: Whether customer has phone service (Yes/No)
            - **Multiple Lines**: Whether customer has multiple phone lines
            - **Internet Service**: Type of internet service (DSL/Fiber optic/No)
            - **Online Security**: Whether customer has online security add-on
            - **Online Backup**: Whether customer has online backup add-on
            """)
        
        with col2:
            st.markdown("""
            **Account Information:**
            - **Tenure**: Number of months customer has stayed with company
            - **Contract**: Contract term (Month-to-month/One year/Two year)
            - **Paperless Billing**: Whether customer uses paperless billing (Yes/No)
            - **Payment Method**: How customer pays their bill
            - **Monthly Charges**: Amount charged to customer monthly
            - **Total Charges**: Total amount charged to customer
            
            **Additional Services:**
            - **Tech Support**: Whether customer has tech support add-on
            - **Device Protection**: Whether customer has device protection add-on
            - **Streaming TV**: Whether customer has streaming TV add-on
            - **Streaming Movies**: Whether customer has streaming movies add-on
            """)

def get_user_input():
    """Create input form for customer data."""
    st.markdown('<div class="main-header"><h1>Customer Churn Prediction App</h1></div>', unsafe_allow_html=True)
    
    # Initialize session state for reset functionality
    if 'reset_form' not in st.session_state:
        st.session_state.reset_form = False
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📝 Manual Input", "📄 Sample Data"])
    
    with tab1:
        # Create form for user input
        with st.form("customer_form"):
            st.subheader("Enter Customer Information")
            
            # Set default values based on reset state
            default_values = {
                'gender': "Male",
                'senior_citizen': 0,
                'partner': "Yes",
                'dependents': "Yes",
                'tenure': 12,
                'phone_service': "Yes",
                'multiple_lines': "No phone service",
                'internet_service': "DSL",
                'online_security': "No internet service",
                'online_backup': "No internet service",
                'device_protection': "No internet service",
                'tech_support': "No internet service",
                'streaming_tv': "No internet service",
                'streaming_movies': "No internet service",
                'contract': "Month-to-month",
                'paperless_billing': "Yes",
                'payment_method': "Electronic check",
                'monthly_charges': 50.0,
                'total_charges': 600.0
            }
            
            # Create columns for input fields
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Demographics**")
                gender = st.selectbox("Gender", ["Male", "Female"], index=0 if default_values['gender'] == "Male" else 1)
                senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=default_values['senior_citizen'])
                partner = st.selectbox("Partner", ["Yes", "No"], index=0 if default_values['partner'] == "Yes" else 1)
                dependents = st.selectbox("Dependents", ["Yes", "No"], index=0 if default_values['dependents'] == "Yes" else 1)
                
                st.markdown("**Services**")
                phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if default_values['phone_service'] == "Yes" else 1)
                multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"], index=0)
                
            with col2:
                st.markdown("**Internet & Add-ons**")
                internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"], index=0)
                online_security = st.selectbox("Online Security", ["No internet service", "No", "Yes"], index=0)
                online_backup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"], index=0)
                device_protection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"], index=0)
                tech_support = st.selectbox("Tech Support", ["No internet service", "No", "Yes"], index=0)
                streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"], index=0)
                streaming_movies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"], index=0)
                
            with col3:
                st.markdown("**Account Information**")
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=default_values['tenure'])
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"], index=0)
                paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
                payment_method = st.selectbox("Payment Method", 
                                            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=0)
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=default_values['monthly_charges'], step=0.01)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=default_values['total_charges'], step=0.01)
            
            # Submit and Reset buttons
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)
            with col_btn2:
                reset_clicked = st.form_submit_button("🔄 Reset Form", use_container_width=True, type="secondary")
            
            # Handle reset
            if reset_clicked:
                # Clear any prediction results from session state
                if 'prediction_results' in st.session_state:
                    del st.session_state.prediction_results
                st.rerun()
            
            if submitted:
                # Create customer data dictionary
                customer_data = {
                    'gender': gender,
                    'SeniorCitizen': senior_citizen,
                    'Partner': partner,
                    'Dependents': dependents,
                    'tenure': tenure,
                    'PhoneService': phone_service,
                    'MultipleLines': multiple_lines,
                    'InternetService': internet_service,
                    'OnlineSecurity': online_security,
                    'OnlineBackup': online_backup,
                    'DeviceProtection': device_protection,
                    'TechSupport': tech_support,
                    'StreamingTV': streaming_tv,
                    'StreamingMovies': streaming_movies,
                    'Contract': contract,
                    'PaperlessBilling': paperless_billing,
                    'PaymentMethod': payment_method,
                    'MonthlyCharges': monthly_charges,
                    'TotalCharges': total_charges
                }
                
                return customer_data
    
    with tab2:
        st.subheader("Select from Sample Data")
        sample_data = load_sample_data()
        
        if sample_data is not None:
            selected_sample = st.selectbox("Choose a sample customer:", 
                                         range(len(sample_data)), 
                                         format_func=lambda x: f"Customer {x+1}: {sample_data.iloc[x]['Contract']} contract, ${sample_data.iloc[x]['MonthlyCharges']}/month")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                predict_sample = st.button("🔮 Predict for Selected Customer", use_container_width=True)
            with col_btn2:
                if st.button("🔄 Reset Selection", use_container_width=True, type="secondary"):
                    st.rerun()
            
            if predict_sample:
                customer_row = sample_data.iloc[selected_sample].to_dict()
                # Remove customerID and Churn columns if they exist
                customer_data = {k: v for k, v in customer_row.items() if k not in ['customerID', 'Churn']}
                return customer_data
            
            # Display selected customer details
            st.write("**Selected Customer Details:**")
            st.dataframe(sample_data.iloc[[selected_sample]], use_container_width=True)
    
    return None

def display_prediction_results(customer_data, predictor):
    """Display prediction results with visualizations."""
    try:
        # Make prediction
        prediction, probabilities = predictor.predict_churn(customer_data)
        
        if prediction is None:
            st.error("Failed to make prediction. Please check your input data.")
            return
        
        churn_probability = probabilities[1]
        no_churn_probability = probabilities[0]
        
        # Create prediction display
        st.markdown("---")
        st.subheader("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Main prediction card
            churn_status = "High Risk" if prediction == 1 else "Low Risk"
            card_class = "churn-high" if prediction == 1 else "churn-low"
            
            st.markdown(f"""
            <div class="prediction-card {card_class}">
                <h3>Customer Churn Risk: {churn_status}</h3>
                <p style="font-size: 1.2em;">
                    This customer is <strong>{"likely to churn" if prediction == 1 else "unlikely to churn"}</strong>
                </p>
                <p>Confidence: {max(churn_probability, no_churn_probability):.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Churn probability gauge
            st.metric(
                label="Churn Probability",
                value=f"{churn_probability:.1%}",
                delta=f"{churn_probability - 0.5:.1%}" if churn_probability > 0.5 else f"{0.5 - churn_probability:.1%}",
                delta_color="inverse"
            )
        
        with col3:
            # Risk level
            if churn_probability >= 0.7:
                risk_level = "🔴 High"
                risk_color = "#ff4444"
            elif churn_probability >= 0.4:
                risk_level = "🟡 Medium"
                risk_color = "#ffaa00"
            else:
                risk_level = "🟢 Low"
                risk_color = "#44ff44"
            
            st.metric(
                label="Risk Level",
                value=risk_level
            )
        
        # Probability visualization
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Probability bar chart
            fig_bar = go.Figure(data=[
                go.Bar(name='Will Churn', x=['Prediction'], y=[churn_probability], marker_color='#ff6b6b'),
                go.Bar(name='Will Stay', x=['Prediction'], y=[no_churn_probability], marker_color='#4ecdc4')
            ])
            fig_bar.update_layout(
                title='Churn Probability',
                yaxis_title='Probability',
                barmode='stack',
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Probability pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Will Stay', 'Will Churn'],
                values=[no_churn_probability, churn_probability],
                marker_colors=['#4ecdc4', '#ff6b6b'],
                hole=0.4
            )])
            fig_pie.update_layout(
                title='Prediction Distribution',
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Customer profile summary
        st.subheader("📋 Customer Profile Summary")
        profile_col1, profile_col2, profile_col3 = st.columns(3)
        
        with profile_col1:
            st.markdown("**Demographics**")
            st.write(f"• Gender: {customer_data.get('gender', 'N/A')}")
            st.write(f"• Senior Citizen: {'Yes' if customer_data.get('SeniorCitizen', 0) == 1 else 'No'}")
            st.write(f"• Partner: {customer_data.get('Partner', 'N/A')}")
            st.write(f"• Dependents: {customer_data.get('Dependents', 'N/A')}")
        
        with profile_col2:
            st.markdown("**Services**")
            st.write(f"• Tenure: {customer_data.get('tenure', 0)} months")
            st.write(f"• Internet Service: {customer_data.get('InternetService', 'N/A')}")
            st.write(f"• Contract: {customer_data.get('Contract', 'N/A')}")
            st.write(f"• Payment Method: {customer_data.get('PaymentMethod', 'N/A')}")
        
        with profile_col3:
            st.markdown("**Financial**")
            monthly_charges = customer_data.get('MonthlyCharges', 0)
            total_charges = customer_data.get('TotalCharges', 0)
            tenure = customer_data.get('tenure', 1)
            
            # Convert to float if they are strings
            try:
                monthly_charges = float(monthly_charges) if monthly_charges else 0
                total_charges = float(total_charges) if total_charges else 0
                tenure = int(tenure) if tenure else 1
            except (ValueError, TypeError):
                monthly_charges = 0
                total_charges = 0
                tenure = 1
            
            st.write(f"• Monthly Charges: ${monthly_charges:.2f}")
            st.write(f"• Total Charges: ${total_charges:.2f}")
            avg_monthly = total_charges / max(tenure, 1)
            st.write(f"• Avg Monthly: ${avg_monthly:.2f}")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        if prediction == 1:
            st.warning("""
            **High Churn Risk - Immediate Action Recommended:**
            - 🎯 Offer personalized retention offers
            - 📞 Schedule a customer success call
            - 💰 Consider loyalty discounts or upgrades
            - 📋 Review contract terms and payment options
            - 🛠️ Improve service quality based on customer needs
            """)
        else:
            st.success("""
            **Low Churn Risk - Maintenance Actions:**
            - ✅ Continue providing excellent service
            - 📈 Consider upselling opportunities
            - 💌 Send appreciation communications
            - 🔄 Monitor for any changes in behavior
            - 🎁 Offer loyalty rewards
            """)
        
    except Exception as e:
        st.error(f"Error displaying results: {e}")

def display_model_insights():
    """Display model insights and SHAP explanations."""
    st.subheader("🧠 Model Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Performance:**")
        st.info("""
        Our churn prediction model uses XGBoost and achieves:
        - High accuracy in identifying at-risk customers
        - Balanced precision and recall scores
        - Reliable probability estimates
        """)
    
    with col2:
        st.markdown("**Key Factors for Churn:**")
        st.info("""
        Top factors that influence churn:
        - Contract type (month-to-month = higher risk)
        - Tenure (newer customers = higher risk)
        - Monthly charges (higher charges = higher risk)
        - Payment method (electronic check = higher risk)
        """)
    
    # Display SHAP plots if available
    if os.path.exists('shap_summary_plot.png'):
        st.subheader("🔍 Feature Importance (SHAP Analysis)")
        st.image('shap_summary_plot.png', caption="SHAP Summary Plot - How each feature impacts churn prediction")
    
    if os.path.exists('xgboost_feature_importance.png'):
        st.subheader("📊 XGBoost Feature Importance")
        st.image('xgboost_feature_importance.png', caption="Top 20 most important features for churn prediction")

def main():
    """Main Streamlit app function."""
    
    # Sidebar for navigation
    with st.sidebar:
        st.title("🎛️ Navigation")
        page = st.radio("Choose a page:", ["🔮 Prediction", "📊 Model Insights", "ℹ️ About"])
        
        st.markdown("---")
        st.markdown("**Model Status:**")
        
        # Check if model exists
        if os.path.exists('models/best_model.pkl'):
            st.success("✅ Model loaded")
        else:
            st.error("❌ Model not found")
            st.info("Please run `python model.py` first to train the model.")
    
    # Load model
    if os.path.exists('models/best_model.pkl'):
        predictor = load_model_and_metadata()
    else:
        predictor = None
    
    # Page routing
    if page == "🔮 Prediction":
        if predictor is None:
            st.error("❌ Model not loaded. Please train the model first by running `python model.py`")
            return
        
        # Feature explanations
        create_feature_explanation()
        
        # Get user input
        customer_data = get_user_input()
        
        # Display prediction if data is provided
        if customer_data:
            display_prediction_results(customer_data, predictor)
    
    elif page == "📊 Model Insights":
        display_model_insights()
    
    elif page == "ℹ️ About":
        st.markdown("""
        # 📈 Customer Churn Prediction App
        
        ## What is Customer Churn?
        Customer churn refers to when customers stop doing business with a company. 
        Predicting churn helps businesses:
        - 🎯 Identify at-risk customers
        - 💰 Reduce revenue loss
        - 📈 Improve customer retention
        - 🔧 Optimize business strategies
        
        ## How This App Works
        1. **Data Input**: Enter customer information or select from samples
        2. **AI Prediction**: Our trained XGBoost model analyzes the data
        3. **Risk Assessment**: Get churn probability and risk level
        4. **Recommendations**: Receive actionable insights for retention
        
        ## Model Details
        - **Algorithm**: XGBoost (Gradient Boosting)
        - **Features**: 19+ customer attributes
        - **Training Data**: Telco Customer Churn Dataset
        - **Evaluation**: Cross-validated with multiple metrics
        
        ## Technical Stack
        - **Backend**: Python, scikit-learn, XGBoost
        - **Frontend**: Streamlit
        - **Explainability**: SHAP (SHapley Additive exPlanations)
        - **Visualization**: Plotly, Matplotlib, Seaborn
        
        ---
        
        **Built with ❤️ for better customer retention strategies**
        """)

if __name__ == "__main__":
    main()
