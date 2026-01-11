"""
User Streamlit App for Telco Customer Churn Prediction

This app provides a user-friendly interface for predicting customer churn
using the Random Forest Classifier.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_data, prepare_data, get_numerical_categorical_cols
from src.models import RFCModel
from src.utils import can_be_number


# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2ecc71;
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 2rem 0;
    }
    .churn-yes {
        background-color: #ffebee;
        color: #c62828;
        border: 3px solid #c62828;
    }
    .churn-no {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 3px solid #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_prepare_data():
    """Load and prepare data with caching."""
    customer_churn = load_data()
    X_train, X_valid, y_train, y_valid, raw_feature_cols = prepare_data(customer_churn)
    numerical_cols, categorical_cols = get_numerical_categorical_cols(X_train)
    return X_train, X_valid, y_train, y_valid, raw_feature_cols, numerical_cols, categorical_cols


@st.cache_resource
def train_model(_X_train, _y_train, numerical_cols, categorical_cols):
    """Train the Random Forest model with caching."""
    model = RFCModel(numerical_cols, categorical_cols)
    model.fit(_X_train, _y_train)
    return model


def get_user_input_streamlit(X_train, raw_feature_cols):
    """
    Create Streamlit input widgets for all features.
    
    Args:
        X_train: Training DataFrame to get possible values from
        raw_feature_cols: List of feature column names
        
    Returns:
        Dictionary of user inputs
    """
    user_data = {}
    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    for idx, col in enumerate(raw_feature_cols):
        # Alternate between columns
        current_col = col1 if idx % 2 == 0 else col2
        
        with current_col:
            if col in numerical_cols or can_be_number(X_train[col]):
                # Numerical input
                min_val = float(X_train[col].min())
                max_val = float(X_train[col].max())
                mean_val = float(X_train[col].mean())
                
                user_data[col] = st.number_input(
                    f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    help=f"Range: {min_val:.2f} - {max_val:.2f}"
                )
            else:
                # Categorical input
                unique_vals = sorted(X_train[col].unique().tolist())
                user_data[col] = st.selectbox(
                    f"{col}",
                    options=unique_vals,
                    help=f"Select one of {len(unique_vals)} options"
                )
    
    return user_data


def main():
    # Header
    st.markdown('<p class="main-header">🎯 Customer Churn Prediction</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Introduction
    st.markdown("""
    ### Welcome! 👋
    
    This tool helps predict whether a customer is likely to **churn** (cancel their subscription) 
    based on their account information and usage patterns.
    
    **How to use:**
    1. Fill in the customer information below
    2. Click "Predict Churn" button
    3. View the prediction results
    """)
    
    # Load data and train model
    with st.spinner("Loading system..."):
        X_train, X_valid, y_train, y_valid, raw_feature_cols, numerical_cols, categorical_cols = load_and_prepare_data()
        model = train_model(X_train, y_train, numerical_cols, categorical_cols)
    
    st.success("✅ System ready!")
    
    st.markdown("---")
    st.header("📝 Enter Customer Information")
    
    # Get user inputs
    user_data = get_user_input_streamlit(X_train, raw_feature_cols)
    
    # Prediction section
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🔮 Predict Churn", type="primary", use_container_width=True)
    
    if predict_button:
        # Create DataFrame from user input
        user_df = pd.DataFrame([user_data])
        
        # Make prediction
        with st.spinner("Analyzing customer data..."):
            prediction = model.predict(user_df)
        
        # Display result
        st.markdown("### Prediction Result:")
        
        if prediction[0] == 1:
            st.markdown(
                '<div class="prediction-box churn-yes">⚠️ Customer is likely to CHURN</div>',
                unsafe_allow_html=True
            )
            st.warning("""
            **Recommendation:** Consider implementing retention strategies for this customer:
            - Offer personalized discounts or promotions
            - Improve customer service engagement
            - Provide loyalty rewards
            - Address service quality concerns
            """)
        else:
            st.markdown(
                '<div class="prediction-box churn-no">✅ Customer is likely to STAY</div>',
                unsafe_allow_html=True
            )
            st.success("""
            **Great news!** This customer shows positive retention indicators:
            - Continue providing excellent service
            - Maintain regular engagement
            - Consider upselling opportunities
            """)
        
        # Show input summary
        with st.expander("📊 View Input Summary"):
            st.dataframe(user_df.T, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**Telco Customer Churn Prediction System** | Powered by Machine Learning")
    
    # Sidebar info
    st.sidebar.header("ℹ️ About")
    st.sidebar.info("""
    **Model:** Random Forest Classifier
    
    **Features:** 
    - Customer demographics
    - Service subscriptions
    - Account information
    - Usage patterns
    
    **Accuracy:** ~80% on validation set
    """)
    
    st.sidebar.header("📖 Feature Descriptions")
    with st.sidebar.expander("Learn more about features"):
        st.markdown("""
        - **MonthlyCharges:** Amount charged per month
        - **TotalCharges:** Total amount charged to date
        - **Tenure:** Number of months as customer
        - **Contract:** Type of contract (Month-to-month, One year, Two year)
        - **PaymentMethod:** How customer pays their bill
        - **InternetService:** Type of internet service
        - And many more...
        """)


if __name__ == "__main__":
    main()
