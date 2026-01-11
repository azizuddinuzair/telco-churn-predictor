"""
Telco Customer Churn Prediction - Console Entry Point

This script serves as the main entry point for the console version of the application.
Users can choose between developer mode and user mode.
"""

from src.data_loader import load_data, prepare_data, get_numerical_categorical_cols
from src.interfaces import run_developer_interface, run_user_interface


def main():
    """Main entry point for the console application."""
    
    # Load and prepare data
    customer_churn = load_data()
    X_train, X_valid, y_train, y_valid, raw_feature_cols = prepare_data(customer_churn)
    numerical_cols, categorical_cols = get_numerical_categorical_cols(X_train)

    print("\n" + "="*60)
    print("Welcome to Customer Churn Prediction!")
    print("="*60 + "\n")

    user = input("Please enter your name: ")

    if user == "Developer":
        run_developer_interface(X_train, X_valid, y_train, y_valid, numerical_cols, categorical_cols)
    else:
        run_user_interface(X_train, X_valid, y_train, y_valid, numerical_cols, categorical_cols, raw_feature_cols, user)


if __name__ == "__main__":
    main()
