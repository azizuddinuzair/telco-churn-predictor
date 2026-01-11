"""
Base Model and Model Classes for Telco Customer Churn Prediction

This module contains the BaseModel parent class and specific model implementations
(RandomForest, KNN, Logistic Regression) with save/load functionality.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from .feature_pipeline import FeaturePipeline


class BaseModel:
    """
    This is a parent class for different ML models. 
    It holds a pipeline that includes feature engineering, preprocessing, and the model itself.
    
    Purpose:
    - This structure helps to avoid data leakage by ensuring that all transformations are applied consistently.
    - It also makes the code more organized and easier to maintain.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        """
        Constructor to initialize the model with a given pipeline.
        """
        self.pipeline = pipeline

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """
        Train the model using the provided data (it should be training data).
        """
        self.pipeline.fit(X, y)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        Returns an array of predicted labels.
        """
        return self.pipeline.predict(X)
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk using pickle.
        
        Args:
            filepath: Path where the model should be saved (should end with .pkl)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "BaseModel":
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model file (.pkl)
            
        Returns:
            An instance of the model class with the loaded pipeline
        """
        with open(filepath, 'rb') as f:
            pipeline = pickle.load(f)
        
        # Create an instance and set the pipeline
        instance = cls.__new__(cls)
        instance.pipeline = pipeline
        
        print(f"Model loaded from {filepath}")
        return instance


class RFCModel(BaseModel):
    """Random Forest Classifier Model"""
    
    def __init__(self, numerical_cols=None, categorical_cols=None) -> None:
        # Define feature engineering
        feature_engineering = FeaturePipeline()

        # Create pipeline with Random Forest
        my_pipeline = Pipeline(steps=[
            ("feature_engineering", feature_engineering),
            ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=7, max_depth=10, random_state=42))
        ])

        super().__init__(my_pipeline)


class KNNModel(BaseModel):
    """K-Nearest Neighbors Classifier Model"""
    
    def __init__(self, numerical_cols=None, categorical_cols=None) -> None:
        # Define feature engineering
        feature_engineering = FeaturePipeline()

        # Create pipeline with KNN
        my_pipeline = Pipeline(steps=[
            ("feature_engineering", feature_engineering),
            ("model", KNeighborsClassifier(n_neighbors=5, weights="distance", metric="manhattan"))
        ])
        
        super().__init__(my_pipeline)


class LRModel(BaseModel):
    """Logistic Regression Classifier Model"""
    
    def __init__(self, numerical_cols=None, categorical_cols=None) -> None:
        # Define feature engineering
        feature_engineering = FeaturePipeline()

        # Create pipeline with Logistic Regression
        my_pipeline = Pipeline(steps=[
            ("feature_engineering", feature_engineering),
            ("model", LogisticRegression(max_iter=1000))
        ])

        super().__init__(my_pipeline)
