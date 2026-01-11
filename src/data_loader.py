"""
Data Loader for Telco Customer Churn Prediction

This module handles loading and preparing the customer churn dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def load_data(data_path: str = None):
    """
    Load the Telco Customer Churn dataset.
    
    Args:
        data_path: Path to the CSV file. If None, uses default path.
        
    Returns:
        DataFrame containing the customer churn data
    """
    if data_path is None:
        # Find project root by looking for the data directory
        # This works whether running from root, src, or streamlit_apps
        current_file = Path(__file__)
        project_root = current_file.parent.parent  # Go up from src/ to project root
        data_path = project_root / "data" / "Telco-Customer-Churn.csv"
    
    print(f"Loading Telco Customer Churn Data from {data_path}...")
    return pd.read_csv(data_path)


def prepare_data(CustomerChurn: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Prepare the data for training by splitting features and target, 
    and creating train/validation splits.
    
    Args:
        CustomerChurn: Raw DataFrame loaded from CSV
        test_size: Proportion of data to use for validation (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (X_train, X_valid, y_train, y_valid, raw_feature_cols)
        - X_train: Training features
        - X_valid: Validation features  
        - y_train: Training target
        - y_valid: Validation target
        - raw_feature_cols: List of original feature column names
    """
    # Customer ID is not useful for prediction; model would memorize IDs
    # Also dropping some features based on analysis
    X = CustomerChurn.drop(["Churn", "customerID", "SeniorCitizen", "Partner", "Dependents"], axis=1)
    y = CustomerChurn.Churn.replace({"Yes": 1, "No": 0})

    raw_feature_cols = X.columns.tolist()  # Used for user input later

    # Train/validation split with stratification (important for imbalanced classification)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, 
        train_size=(1 - test_size), 
        random_state=random_state, 
        stratify=y
    )

    return X_train, X_valid, y_train, y_valid, raw_feature_cols


def get_numerical_categorical_cols(X: pd.DataFrame):
    """
    Get lists of numerical and categorical column names.
    
    Args:
        X: DataFrame to analyze
        
    Returns:
        Tuple of (numerical_cols, categorical_cols)
    """
    numerical_cols = [col for col in X.columns if X[col].dtype in ["int64", "float64"]]
    categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
    
    return numerical_cols, categorical_cols
