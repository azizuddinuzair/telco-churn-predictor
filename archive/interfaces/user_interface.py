"""
User Console Interface for Telco Customer Churn Prediction

This module provides the user interface for making churn predictions
using the Random Forest Classifier.
"""

from src.models import RFCModel
from src.utils import get_user_data


def run_user_interface(X_train, X_valid, y_train, y_valid, numerical_cols, categorical_cols, raw_feature_cols, user_name):
    """
    Run the user interface for making churn predictions.
    
    Args:
        X_train: Training features
        X_valid: Validation features
        y_train: Training target
        y_valid: Validation target
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
        raw_feature_cols: List of raw feature column names for user input
        user_name: Name of the user
    """
    print(f"Welcome, {user_name}!\n")
    print("We'll be using the Random Forest Classifier for predictions.\n")
    
    # Train the model
    print("Training the model...")
    rfc_model = RFCModel(numerical_cols, categorical_cols)
    rfc_model.fit(X_train, y_train)
    print("Model trained successfully!\n")

    while True:
        print("Please provide the following customer information for churn prediction:")

        userData = get_user_data(X_train, raw_feature_cols)
        prediction = rfc_model.predict(userData)

        if prediction[0] == 1:
            print("\n" + "="*50)
            print("Prediction: The customer is likely to CHURN.")
            print("="*50 + "\n")
        else:
            print("\n" + "="*50)
            print("Prediction: The customer is likely to STAY.")
            print("="*50 + "\n")

        continue_input = input("Would you like to predict for another customer? (y/n): ")
        if continue_input.lower() != "y":
            break

    print("\nExiting User Interface. Goodbye!")
