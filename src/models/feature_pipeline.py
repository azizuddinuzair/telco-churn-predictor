"""
Feature Pipeline for Telco Customer Churn Prediction

This module contains the FeaturePipeline class that handles feature engineering
and preprocessing for the customer churn prediction models.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer.
    
    This class inherits from:
    - BaseEstimator: This helps make the class compatible with scikit-learn's estimator API.
    - TransformerMixin: This provides the fit_transform method, which combines fit and transform.

    Purpose:
    - This helps in creating new features or modifying existing ones in the dataset.
    - It stops the risk of data leakage by ensuring that feature engineering is done within the pipeline.

    BaseEstimator: makes the class parameters easily accessible and allows for hyperparameter tuning.
    TransformerMixin: provides the fit_transform method for convenience.
    """
    
    def __init__(self) -> None:
        """
        Constructor to initialize any parameters if needed.
        """
        self.col_to_drop_ = ["Contract", "tenure", "tenure_bin",
                            "OnlineSecurity", "TechSupport",
                            "OnlineBackup", "DeviceProtection",
                            "PaperlessBilling", "PaymentMethod",
                            "gender", "PhoneService", "MultipleLines"]  # Removing columns with low MI scores

    def fit(self, X: pd.DataFrame, y=None) -> "FeaturePipeline":
        """
        Fit step to learn any parameters from the training data.
        """
        X_fe = self._feature_engineering(X)

        self.numerical_cols_ = [col for col in X_fe.columns if X_fe[col].dtype in ["int64", "float64"]]
        self.categorical_cols_ = [col for col in X_fe.columns if X_fe[col].dtype == "object"]
        
        # Do all preprocessing after feature engineering
        numerical_transformer_ = Pipeline(steps=[
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler())
        ])

        categorical_transformer_ = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("OrdinalEncoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        ])
        
        self.preprocessor_ = ColumnTransformer(transformers=[
            ("num", numerical_transformer_, self.numerical_cols_),
            ("cat", categorical_transformer_, self.categorical_cols_)
        ])

        self.preprocessor_.fit(X_fe)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the feature transformations to the data.
        """
        X_fe = self._feature_engineering(X)
        X_transformed = self.preprocessor_.transform(X_fe)
        
        return pd.DataFrame(X_transformed, 
                          columns=self.numerical_cols_ + self.categorical_cols_, 
                          index=X.index)

    def _feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies only the feature engineering steps without preprocessing.
        Useful for MI score calculations.
        """
        X = X.copy()  # Create a copy to avoid modifying the original data

        # Ensure TotalCharges is numerical so it is treated as a continuous feature
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
            X["TotalCharges"].fillna(X["TotalCharges"].median(), inplace=True)

        # Feature engineering steps
        X["tenure_bin"] = pd.cut(X["tenure"], bins=[0, 6, 12, 24, 48, 75], labels=False, include_lowest=True)
        X["Contract_Tenure"] = X["Contract"].astype(str) + "_" + X["tenure_bin"].astype(str)
        X["OnlineSecurity_TechSupport"] = X["OnlineSecurity"].astype(str) + "_" + X["TechSupport"].astype(str)
        X["OnlineBackup_DeviceProtection"] = X["OnlineBackup"].astype(str) + "_" + X["DeviceProtection"].astype(str)
        X["PaperlessBilling_PaymentMethod"] = X["PaperlessBilling"].astype(str) + "_" + X["PaymentMethod"].astype(str)
        
        # Drop the original columns that were combined
        X = X.drop(self.col_to_drop_, axis=1)

        return X
