"""
Developer Console Interface for Telco Customer Churn Prediction

This module provides the developer interface with advanced features like
mutual information analysis, model comparison, and cross-validation.
"""

from src.models import RFCModel, KNNModel, LRModel
from src.models.feature_pipeline import FeaturePipeline
from src.utils import plot_mi_scores, make_mi_scores, get_score, get_cv_score


def run_developer_interface(X_train, X_valid, y_train, y_valid, numerical_cols, categorical_cols):
    """
    Run the developer interface with advanced model testing and analysis features.
    
    Args:
        X_train: Training features
        X_valid: Validation features
        y_train: Training target
        y_valid: Validation target
        numerical_cols: List of numerical column names
        categorical_cols: List of categorical column names
    """
    model_registry = {}
    model_full_name_registry = {
        "RFC": "Random Forest Classifier",
        "KNN": "K-Nearest Neighbors Classifier",
        "LR": "Logistic Regression Classifier"
    }
    mi_registry = {}
    cv_registry = {}

    print("Welcome, Developer! You have developer access.\n")
    print("Menu Options:")
    print("1. See Mutual Information Scores")
    print("2. See MI Plot")
    print("3. Test Random Forest Classifier")
    print("4. Test K-Nearest Neighbors Classifier")
    print("5. Test Logistic Regression Classifier")
    print("6. Exit")

    while (c := input("\nEnter your choice (1-6): ")) != "6":
        match c:
            case "1" | "2":
                print()
                if "mi_scores" not in mi_registry:
                    feature_pipeline = FeaturePipeline()
                    X_train_fe = feature_pipeline.fit_transform(X_train, y_train)

                    discrete_features = [col in feature_pipeline.categorical_cols_ for col in X_train_fe.columns]
                    mi_scores = make_mi_scores(X_train_fe, y_train, discrete_features)

                    mi_registry["mi_scores"] = mi_scores
                    mi_registry["X_train_fe"] = X_train_fe
                    mi_registry["feature_pipeline"] = feature_pipeline
                else:
                    mi_scores = mi_registry["mi_scores"]

                if c == "1":
                    print("Mutual Information Scores:")
                    print(mi_scores)
                else:
                    plot_mi_scores(mi_scores)
    
            case "3" | "4" | "5":
                if c == "3":
                    model_name = "RFC"
                    model_class = RFCModel
                elif c == "4":
                    model_name = "KNN"
                    model_class = KNNModel
                else:
                    model_name = "LR"
                    model_class = LRModel

                if model_name not in model_registry:
                    print(f"\nTraining {model_full_name_registry[model_name]}...")
                    model = model_class(numerical_cols, categorical_cols)
                    model.fit(X_train, y_train)
                    model_registry[model_name] = model
                else:
                    model = model_registry[model_name]

                score = get_score(model, X_valid, y_valid)

                print(f"\n{model_full_name_registry[model_name]}")
                print("------------------------------")
                print("Evaluation on Validation Set:")
                print(f"F1 Score: {score[0]:.4f}")
                print(f"Accuracy Score: {score[1]:.4f}\n")

                see_cv = input("Would you like to see Cross-Validation scores? (y/n): ")
                if see_cv.lower() == "y":
                    if model_name not in cv_registry:
                        print("Performing 5-Fold Cross-Validation...")
                        cv_score = get_cv_score(model, X_train, y_train)
                        cv_registry[model_name] = cv_score
                    else:
                        cv_score = cv_registry[model_name]
                    
                    print(f"Cross-Validation F1 Score: {cv_score[0]:.4f}")
                    print(f"Cross-Validation Accuracy Score: {cv_score[1]:.4f}\n")
                
            case _:
                print("Invalid choice. Please enter a number between 1 and 6.")
                continue
        
        print("\nMenu Options:")
        print("1. See Mutual Information Scores")
        print("2. See MI Plot")
        print("3. Test Random Forest Classifier")
        print("4. Test K-Nearest Neighbors Classifier")
        print("5. Test Logistic Regression Classifier")
        print("6. Exit")

    print("\nExiting Developer Interface. Goodbye!")
