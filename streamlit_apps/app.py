"""
Combined Streamlit App with Sidebar Navigation

Sidebar options:
- User
- Developer Tools (title)
  - Mutual Information Analysis
  - Model Testing
  - Model Comparison
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Ensure src/ is importable when running from streamlit_apps/
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_data, prepare_data, get_numerical_categorical_cols
from src.models import RFCModel, KNNModel, LRModel
from src.models.feature_pipeline import FeaturePipeline
from src.utils import make_mi_scores, get_score, get_cv_score, can_be_number


# Page configuration
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# Helper functions
def get_user_input_streamlit(X_train, raw_feature_cols):
    """Collect user input via Streamlit form."""
    user_data = {}
    
    # Convert columns to numeric where possible
    numeric_convertible = {}
    for col in raw_feature_cols:
        col_data = X_train[col]
        try:
            converted = pd.to_numeric(col_data, errors='coerce')
            if converted.notna().sum() / len(converted) >= 0.9:
                numeric_convertible[col] = True
                col_data = converted.dropna()
            else:
                numeric_convertible[col] = False
        except:
            numeric_convertible[col] = False
    
    # Create input fields
    cols = st.columns(3)
    for idx, col in enumerate(raw_feature_cols):
        with cols[idx % 3]:
            if numeric_convertible.get(col, False):
                col_data = pd.to_numeric(X_train[col], errors='coerce').dropna()
                min_val = float(col_data.min())
                max_val = float(col_data.max())
                mean_val = float(col_data.mean())
                user_data[col] = st.number_input(
                    col,
                    value=mean_val,
                    help=f"Training range: {min_val:.2f} - {max_val:.2f}",
                    key=f"user_input_{col}"
                )
            else:
                options = sorted(X_train[col].unique().tolist())
                user_data[col] = st.selectbox(col, options, key=f"user_select_{col}")
    
    return user_data


@st.cache_data
def compute_mi_scores(_X_train, _y_train):
    """Compute MI scores with feature engineering."""
    feature_pipeline = FeaturePipeline()
    X_train_fe = feature_pipeline.fit_transform(_X_train, _y_train)
    # Identify discrete columns (categorical or integer types after encoding)
    discrete_cols = X_train_fe.dtypes != float
    mi_scores = make_mi_scores(X_train_fe, _y_train, discrete_cols)
    return mi_scores, X_train_fe, feature_pipeline


@st.cache_data
def load_and_prepare_data():
    """Shared data loader for both tabs."""
    customer_churn = load_data()
    X_train, X_valid, y_train, y_valid, raw_feature_cols = prepare_data(customer_churn)
    numerical_cols, categorical_cols = get_numerical_categorical_cols(X_train)
    return X_train, X_valid, y_train, y_valid, raw_feature_cols, numerical_cols, categorical_cols


@st.cache_resource
def train_model(model_key, X_train, y_train, numerical_cols, categorical_cols):
    """Train and cache model."""
    model_map = {
        "RFC": RFCModel,
        "KNN": KNNModel,
        "LR": LRModel
    }
    model = model_map[model_key](numerical_cols, categorical_cols)
    model.fit(X_train, y_train)
    return model


# Sidebar Navigation
st.sidebar.title("Navigation")

# User section
if st.sidebar.button("User", width='stretch', key="nav_user"):
    st.session_state.page = "User"

# Developer Tools section
st.sidebar.markdown("### Developer Tools")
if st.sidebar.button("Mutual Information", width='stretch', key="nav_mi"):
    st.session_state.page = "MI Analysis"
if st.sidebar.button("Model Testing", width='stretch', key="nav_test"):
    st.session_state.page = "Model Testing"
if st.sidebar.button("Model Comparison", width='stretch', key="nav_compare"):
    st.session_state.page = "Model Comparison"

# Initialize page state
if 'page' not in st.session_state:
    st.session_state.page = "User"

page = st.session_state.page

# Load data once
X_train, X_valid, y_train, y_valid, raw_feature_cols, numerical_cols, categorical_cols = load_and_prepare_data()


# ------------------------------
# User Page
# ------------------------------
if page == "User":
    st.header("Customer Churn Prediction - User")
    st.write("Fill in customer details and predict churn in real time.")

    with st.spinner("Loading model..."):
        model = train_model("RFC", X_train, y_train, numerical_cols, categorical_cols)

    st.success("System ready!")

    st.markdown("---")
    st.subheader("Enter Customer Information")

    # Inputs
    user_data_dict = get_user_input_streamlit(X_train, raw_feature_cols)
    user_df = pd.DataFrame([user_data_dict])

    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("Predict Churn", type="primary", width='stretch', key="user_predict")

    if predict_button:
        with st.spinner("Analyzing customer data..."):
            prediction = model.predict(user_df)

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("Customer is likely to CHURN")
        else:
            st.success("Customer is likely to STAY")

        with st.expander("View Input Summary"):
            st.dataframe(user_df.T, width='stretch')


# ------------------------------
# Mutual Information Analysis
# ------------------------------
elif page == "MI Analysis":
    st.header("Mutual Information Analysis")
    st.write("Analyze feature importance using mutual information scores.")

    if st.button("Compute MI Scores", type="primary", key="dev_compute_mi"):
        with st.spinner("Computing mutual information scores..."):
            mi_scores, X_train_fe, feature_pipeline = compute_mi_scores(X_train, y_train)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("MI Scores Table")
            st.dataframe(mi_scores, height=400)
        with col2:
            st.subheader("Feature Importance Visualization")
            fig, ax = plt.subplots(figsize=(10, 8))
            mi_scores.plot(kind='barh', ax=ax)
            ax.set_xlabel("MI Score")
            ax.set_ylabel("Features")
            ax.set_title("Mutual Information Scores")
            st.pyplot(fig)


# ------------------------------
# Model Testing
# ------------------------------
elif page == "Model Testing":
    st.header("Model Testing")
    st.write("Train and evaluate individual models.")

    model_choice = st.selectbox(
        "Select Model to Test:",
        ["Random Forest Classifier", "K-Nearest Neighbors", "Logistic Regression"],
        key="dev_model_select"
    )
    model_map = {
        "Random Forest Classifier": "RFC",
        "K-Nearest Neighbors": "KNN",
        "Logistic Regression": "LR"
    }
    model_key = model_map[model_choice]

    # Persist trained model and metrics
    if 'dev_model' not in st.session_state:
        st.session_state.dev_model = None
        st.session_state.dev_val_scores = None
        st.session_state.dev_model_name = None

    if st.button(f"Train & Evaluate {model_choice}", type="primary", key="dev_train_eval"):
        with st.spinner(f"Training {model_choice}..."):
            model = train_model(model_key, X_train, y_train, numerical_cols, categorical_cols)
        st.session_state.dev_model = model
        st.session_state.dev_model_name = model_choice
        with st.spinner("Evaluating on validation set..."):
            f1, accuracy = get_score(model, X_valid, y_valid)
        st.session_state.dev_val_scores = (f1, accuracy)
        st.success(f"{model_choice} trained and evaluated!")

    if st.session_state.dev_model is not None:
        st.markdown("---")
        st.subheader(f"Results for {st.session_state.dev_model_name}")
        
        st.subheader("Validation Set Performance")
        col1, col2 = st.columns(2)
        col1.metric("F1 Score", f"{st.session_state.dev_val_scores[0]:.4f}")
        col2.metric("Accuracy", f"{st.session_state.dev_val_scores[1]:.4f}")

        if st.checkbox("Show Cross-Validation Results", key="dev_show_cv"):
            with st.spinner("Performing 5-fold cross-validation... This may take a minute."):
                cv_f1, cv_acc = get_cv_score(st.session_state.dev_model, X_train, y_train)
            st.subheader("Cross-Validation Performance (5-Fold)")
            c1, c2 = st.columns(2)
            c1.metric("CV F1 Score", f"{cv_f1:.4f}")
            c2.metric("CV Accuracy", f"{cv_acc:.4f}")


# ------------------------------
# Model Comparison
# ------------------------------
elif page == "Model Comparison":
    st.header("Model Comparison")
    st.write("Train and compare all models side-by-side.")

    if st.button("Train & Compare All Models", type="primary", key="dev_compare_all"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        models = [
            ("RFC", "Random Forest Classifier"),
            ("KNN", "K-Nearest Neighbors"),
            ("LR", "Logistic Regression")
        ]
        for idx, (m_key, m_name) in enumerate(models):
            status_text.text(f"Training {m_name}...")
            model = train_model(m_key, X_train, y_train, numerical_cols, categorical_cols)
            status_text.text(f"Evaluating {m_name}...")
            f1, accuracy = get_score(model, X_valid, y_valid)
            status_text.text(f"Cross-validating {m_name}...")
            cv_f1, cv_acc = get_cv_score(model, X_train, y_train)
            results.append({
                "Model": m_name,
                "Validation F1": f1,
                "Validation Accuracy": accuracy,
                "CV F1": cv_f1,
                "CV Accuracy": cv_acc
            })
            progress_bar.progress((idx + 1) / len(models))
        status_text.text("All models trained and evaluated!")
        
        st.markdown("---")
        st.subheader("Model Comparison Results")
        results_df = pd.DataFrame(results)
        st.dataframe(
            results_df.style.highlight_max(
                axis=0, 
                subset=["Validation F1", "Validation Accuracy", "CV F1", "CV Accuracy"]
            ),
            width='stretch'
        )
        
        st.subheader("Performance Visualizations")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        results_df.plot(x='Model', y=['Validation F1', 'Validation Accuracy'], kind='bar', ax=axes[0], rot=30)
        axes[0].set_title("Validation Set Performance")
        axes[0].set_ylabel("Score")
        results_df.plot(x='Model', y=['CV F1', 'CV Accuracy'], kind='bar', ax=axes[1], rot=30)
        axes[1].set_title("Cross-Validation Performance")
        axes[1].set_ylabel("Score")
        plt.tight_layout()
        st.pyplot(fig)
