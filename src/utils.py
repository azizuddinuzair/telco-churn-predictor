"""
Utility functions for Telco Customer Churn Prediction

This module contains helper functions for model evaluation, feature analysis,
and user input handling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score


def plot_mi_scores(mi_scores: pd.Series) -> None:
    """
    Plot mutual information scores as a bar chart.
    
    Args:
        mi_scores: Series containing MI scores with feature names as index
    """
    plt.figure(dpi=100, figsize=(12, 8))
    sns.barplot(x=mi_scores.values, y=mi_scores.index)
    plt.title("Mutual Information Scores")
    plt.xlabel("MI Score")
    plt.ylabel("Features")
    plt.show()


def make_mi_scores(X: pd.DataFrame, y: pd.Series, discrete_cols: pd.Series) -> pd.Series:
    """
    Calculate mutual information scores for features.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        discrete_cols: Boolean series indicating which columns are discrete
        
    Returns:
        Series of MI scores sorted in descending order
    """
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_cols)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def get_score(model: object, X_valid: pd.DataFrame, y_valid: pd.Series) -> tuple:
    """
    Get F1 and accuracy scores for a model on validation data.
    
    Args:
        model: Trained model with predict method
        X_valid: Validation features
        y_valid: Validation target
        
    Returns:
        Tuple of (f1_score, accuracy_score)
    """
    preds = model.predict(X_valid)
    return (f1_score(y_valid, preds), accuracy_score(y_valid, preds))


def get_cv_score(model: object, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    """
    Get cross-validation F1 and accuracy scores for a model.
    
    Args:
        model: Model with pipeline attribute
        X_train: Training features
        y_train: Training target
        
    Returns:
        Tuple of (cv_f1_score, cv_accuracy_score)
    """
    f1_cv_score = cross_val_score(model.pipeline, X_train, y_train, cv=5, scoring="f1").mean()
    acc_cv_score = cross_val_score(model.pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()
    return (f1_cv_score, acc_cv_score)


def can_be_number(s: pd.Series, threshold=0.9) -> bool:
    """
    Helper function to check if a pandas Series can be treated as numerical.
    
    It converts the Series to numeric, coercing errors to NaN, and then checks 
    the proportion of non-NaN values.
    
    Args:
        s: Pandas Series to check
        threshold: Minimum proportion of valid numbers (default: 0.9)
        
    Returns:
        True if the proportion of non-NaN values is >= threshold
    """
    converted = pd.to_numeric(s, errors='coerce')
    return converted.notna().mean() >= threshold


def get_user_data(X_train: pd.DataFrame, raw_feature_cols: list) -> pd.DataFrame:
    """
    Function to get user input for prediction.
    
    Handles both numerical and categorical inputs, provides feedback on invalid inputs,
    and shows possible values for categorical columns.
    
    Args:
        X_train: Training DataFrame to get possible values from
        raw_feature_cols: List of feature column names to collect
        
    Returns:
        DataFrame with single row of user input data
    """
    user_data = {}
    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()

    for col in raw_feature_cols:
        # Show possible values for categorical columns
        if col not in numerical_cols and not can_be_number(X_train[col]):
            unique_vals = X_train[col].unique().tolist()
            print(f"\nPossible values for {col}: {unique_vals}")
        
        # Keep asking until valid input
        while True:
            user_input = input(f"Enter value for {col}: ")
            print()  # For better readability
            
            if col in numerical_cols:
                try:
                    user_data[col] = float(user_input)
                    break
                except ValueError:
                    print(f"Invalid input. Please enter a numerical value for {col}.")
            
            elif can_be_number(X_train[col]):
                try:
                    user_data[col] = float(user_input)
                    break
                except ValueError:
                    print(f"Invalid input. Please enter a numerical value for {col}.")
            else:
                if user_input in X_train[col].values:
                    user_data[col] = user_input
                    break
                else:
                    print(f"Invalid input. Please enter one of the possible values for {col}.")

    return pd.DataFrame([user_data])
