"""
Developer Streamlit App for Telco Customer Churn Prediction

This app provides a developer interface with advanced features:
- Mutual Information Analysis
- Model Comparison (RFC, KNN, Logistic Regression)
- Cross-Validation Results
- Model Performance Metrics
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import load_data, prepare_data, get_numerical_categorical_cols
from src.models import RFCModel, KNNModel, LRModel
from src.models.feature_pipeline import FeaturePipeline
from src.utils import make_mi_scores, get_score, get_cv_score, plot_mi_scores


# Page configuration
st.set_page_config(
    page_title="Developer - Churn Prediction",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
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


@st.cache_data
def compute_mi_scores(_X_train, _y_train):
    """Compute mutual information scores with caching."""
    feature_pipeline = FeaturePipeline()
    X_train_fe = feature_pipeline.fit_transform(_X_train)
    discrete_features = [col in feature_pipeline.categorical_cols_ for col in X_train_fe.columns]
    mi_scores = make_mi_scores(X_train_fe, _y_train, discrete_features)
    return mi_scores, X_train_fe, feature_pipeline


@st.cache_resource
def train_model(model_name, _X_train, _y_train, numerical_cols, categorical_cols):
    """Train a model with caching."""
    if model_name == "RFC":
        model = RFCModel(numerical_cols, categorical_cols)
    elif model_name == "KNN":
        model = KNNModel(numerical_cols, categorical_cols)
    else:
        model = LRModel(numerical_cols, categorical_cols)
    
    model.fit(_X_train, _y_train)
    return model


def main():
    # Header
    st.markdown('<p class="main-header">🔬 Developer Interface - Customer Churn Prediction</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        X_train, X_valid, y_train, y_valid, raw_feature_cols, numerical_cols, categorical_cols = load_and_prepare_data()
    
    st.success(f"Data loaded: {len(X_train)} training samples, {len(X_valid)} validation samples")
    
    # Sidebar
    st.sidebar.header("Developer Tools")
    analysis_type = st.sidebar.radio(
        "Select Analysis Type:",
        ["Mutual Information Analysis", "Model Testing", "Model Comparison"]
    )
    
    # Main content based on selection
    if analysis_type == "Mutual Information Analysis":
        st.header("Mutual Information Analysis")
        st.write("Mutual Information (MI) measures the dependency between features and the target variable.")
        
        if st.button("Compute MI Scores", type="primary"):
            with st.spinner("Computing mutual information scores..."):
                mi_scores, X_train_fe, feature_pipeline = compute_mi_scores(X_train, y_train)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("MI Scores Table")
                st.dataframe(mi_scores, height=400)
            
            with col2:
                st.subheader("MI Scores Visualization")
                fig, ax = plt.subplots(figsize=(10, 8))
                mi_scores.plot(kind='barh', ax=ax)
                ax.set_xlabel("MI Score")
                ax.set_ylabel("Features")
                ax.set_title("Mutual Information Scores")
                st.pyplot(fig)
    
    elif analysis_type == "Model Testing":
        st.header("Model Testing")
        
        model_choice = st.selectbox(
            "Select Model to Test:",
            ["Random Forest Classifier", "K-Nearest Neighbors", "Logistic Regression"]
        )
        
        model_map = {
            "Random Forest Classifier": "RFC",
            "K-Nearest Neighbors": "KNN",
            "Logistic Regression": "LR"
        }
        model_name = model_map[model_choice]
        
        # Initialize session state for storing model results
        if 'trained_model' not in st.session_state:
            st.session_state.trained_model = None
            st.session_state.model_name = None
            st.session_state.validation_scores = None
        
        if st.button(f"Train & Evaluate {model_choice}", type="primary"):
            with st.spinner(f"Training {model_choice}..."):
                model = train_model(model_name, X_train, y_train, numerical_cols, categorical_cols)
            
            # Store in session state
            st.session_state.trained_model = model
            st.session_state.model_name = model_choice
            
            st.success(f"✅ {model_choice} trained successfully!")
            
            # Validation scores
            with st.spinner("Evaluating on validation set..."):
                f1, accuracy = get_score(model, X_valid, y_valid)
            
            # Store validation scores
            st.session_state.validation_scores = (f1, accuracy)
        
        # Display validation scores if model has been trained
        if st.session_state.trained_model is not None:
            st.subheader("Validation Set Performance")
            col1, col2 = st.columns(2)
            col1.metric("F1 Score", f"{st.session_state.validation_scores[0]:.4f}")
            col2.metric("Accuracy", f"{st.session_state.validation_scores[1]:.4f}")
            
            # Cross-validation option
            if st.checkbox("Show Cross-Validation Results"):
                with st.spinner("⏳ Performing 5-fold cross-validation... This may take a minute."):
                    cv_f1, cv_acc = get_cv_score(st.session_state.trained_model, X_train, y_train)
                
                st.subheader("Cross-Validation Performance (5-Fold)")
                col1, col2 = st.columns(2)
                col1.metric("CV F1 Score", f"{cv_f1:.4f}")
                col2.metric("CV Accuracy", f"{cv_acc:.4f}")
    
    else:  # Model Comparison
        st.header("Model Comparison")
        st.write("Compare performance across all models")
        
        if st.button("Train & Compare All Models", type="primary"):
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            models = [
                ("RFC", "Random Forest Classifier"),
                ("KNN", "K-Nearest Neighbors"),
                ("LR", "Logistic Regression")
            ]
            
            for idx, (model_name, model_full_name) in enumerate(models):
                status_text.text(f"🔄 Training {model_full_name}...")
                
                model = train_model(model_name, X_train, y_train, numerical_cols, categorical_cols)
                
                status_text.text(f"📊 Evaluating {model_full_name}...")
                f1, accuracy = get_score(model, X_valid, y_valid)
                
                status_text.text(f"⏳ Running cross-validation for {model_full_name}... Please wait.")
                cv_f1, cv_acc = get_cv_score(model, X_train, y_train)
                
                results.append({
                    "Model": model_full_name,
                    "Validation F1": f1,
                    "Validation Accuracy": accuracy,
                    "CV F1": cv_f1,
                    "CV Accuracy": cv_acc
                })
                
                progress_bar.progress((idx + 1) / len(models))
            
            status_text.text("All models trained!")
            
            # Display results
            results_df = pd.DataFrame(results)
            st.subheader("Comparison Results")
            st.dataframe(results_df.style.highlight_max(axis=0, subset=["Validation F1", "Validation Accuracy", "CV F1", "CV Accuracy"]), 
                        use_container_width=True)
            
            # Visualization
            st.subheader("Performance Comparison")
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Validation scores
            results_df.plot(x='Model', y=['Validation F1', 'Validation Accuracy'], 
                          kind='bar', ax=axes[0], rot=45)
            axes[0].set_title("Validation Set Performance")
            axes[0].set_ylabel("Score")
            axes[0].legend(["F1 Score", "Accuracy"])
            
            # CV scores
            results_df.plot(x='Model', y=['CV F1', 'CV Accuracy'], 
                          kind='bar', ax=axes[1], rot=45)
            axes[1].set_title("Cross-Validation Performance")
            axes[1].set_ylabel("Score")
            axes[1].legend(["CV F1", "CV Accuracy"])
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Footer
    st.markdown("---")
    st.markdown("**Developer Mode** | Telco Customer Churn Prediction System")


if __name__ == "__main__":
    main()
