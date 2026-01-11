# Problem: 
# You are given customer data from a subscription-based company. 
# Your task is to predict whether a customer will churn (cancel their subscription) 
# based on their usage patterns, account information, and demographics.
# This is a binary classification problem.


"""
Name: Uzair Azizuddin
Date: 2025-12-31
Project: Customer Churn Prediction for Telco Company
Goal: Build a machine learning model to predict customer churn using feature engineering and various classifiers.

How I did it:

First, I created a FeaturePipeline class to handle feature engineering and preprocessing.
This class includes methods to create new features, modify existing ones, and drop unnecessary columns.
I then defined a BaseModel class to encapsulate the machine learning model along with the feature pipeline.
This class is a parent class that ensures that all transformations are applied consistently and helps prevent data leakage.
Child classes for specific models (Random Forest, KNN, Logistic Regression) inherit from BaseModel.

The main script loads the customer churn data, splits it into training and validation sets, and allows for interactive testing of different models.
If the user is a developer, they can view mutual information scores and test different models by typing in "Developer" as their name.

Otherwise, regular users can input customer data to get churn predictions using the Random Forest Classifier.

Issues that I encountered and fixed:
1. Leakage Issues Fixed: The previous implementation had significant data leakage issues due to feature engineering and preprocessing being done outside the model pipeline. This has been resolved by integrating these steps into a single pipeline.
2. Categorical Encoding: The previous manual factorization of categorical features caused issues with unseen categories in the validation set. This has been addressed by using OrdinalEncoder within the pipeline.
3. Mutual Information Scores: Some combined features significantly improved MI scores, such as Contract_Tenure and PaperlessBilling_PaymentMethod. These combinations have been retained.
4. Consistent Preprocessing: All models now use the same preprocessing steps, ensuring consistency and reducing discrepancies between cross-validation and validation performance.

Observations:
1. The Random Forest Classifier performed the best among the tested models, achieving the highest F1 and accuracy scores on the validation set.
    It also had the most stable cross-validation scores, showing that the quality of the model is consistent across different data splits.
2. K-Nearest Neighbors initially performed poorly due to scaling issues, and even after tuning the hyperparameters, it still lagged behind the Random Forest.
3. Logistic Regression initially showed the best performance and quality at the start, so I decided to keep it as a benchmark model for comparison. 
I'm sure with further tuning, it could perform better than RFC.

"""


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import sklearn

#Preprocessing imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

#Model imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# #Evaluation imports
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score




# This class manages raw features and turns them into engineered features
# fit method learns any parameters needed for transformation from the training data
# transform method applies the learned transformations to the data
class FeaturePipeline(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer.
    
    This class inherirt from:
    - BaseEstimator: This helps make the class compatible with scikit-learn's estimator API.
    - TransformerMixin: This provides the fit_transform method, which combines fit and transform.

    Purpose:
    - This helps in creating new features or modifying existing ones in the dataset.
    - It stops the risk of data leakage by ensuring that feature engineering is done within the pipeline.

    BaseEstimator: makes the class parameters easily accessible and allows for hyperparameter tuning.
    TransformerMixin: provides the fit_transform method for convenience.
    
    """
    def __init__(self) -> None: # We pass the columns to the constuctor because we need to know which columns are numerical or categorical for feature engineering.
        """
        Constructor to initialize any parameters if needed.
        """
        self.col_to_drop_ = ["Contract","tenure", "tenure_bin",
                    "OnlineSecurity", "TechSupport",
                    "OnlineBackup", "DeviceProtection",
                    "PaperlessBilling", "PaymentMethod",
                    "gender", "PhoneService", "MultipleLines"]  # Removing columns with low MI scores        




    def fit(self, X: pd.DataFrame, y = None) -> "FeaturePipeline":
        """
        Fit step to learn any parameters from the training data.
        """
        # feature engineering steps go here
        # Create new Columns
        # Modify existing Columns
        # Drop unnecessary Columns

        X_fe = self._feature_engineering(X)

    
            
        self.numerical_cols_ = [col for col in X_fe.columns if X_fe[col].dtype in ["int64", "float64"]]
        self.categorical_cols_ = [col for col in X_fe.columns if X_fe[col].dtype == "object"]
        #Do all preprocessing after feature engineering
        numerical_transformer_ = Pipeline(steps=[
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler())
        ])

        categorical_transformer_ = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")), #Problem #2 FIXED: Changed strategy to most_frequent for categorical data (before I didn't have anything so it defaulted to mean, which is invalid for categorical data)
            # ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
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

        # return pd.DataFrame(X_transformed, columns=self.preprocessor_.get_feature_names_out(), index=X.index) # This is for OneHotEncoder
        return pd.DataFrame(X_transformed, columns= self.numerical_cols_ + self.categorical_cols_, index=X.index) # This is for OrdinalEncoder


    
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
            # before binning: MI score = 0.039958 
            # after binning: MI score = 0.038678 (not significant enough to keep)
            # X["TotalCharges_bin"] = pd.cut(X["TotalCharges"], bins=10, labels=False, include_lowest=True)

        # feature engineering steps go here
        X["tenure_bin"] = pd.cut(X["tenure"], bins=[0, 6, 12, 24, 48, 75], labels=False, include_lowest=True)
        X["Contract_Tenure"] = X["Contract"].astype(str) + "_" + X["tenure_bin"].astype(str)
        X["OnlineSecurity_TechSupport"] = X["OnlineSecurity"].astype(str) + "_" + X["TechSupport"].astype(str)
        X["OnlineBackup_DeviceProtection"] = X["OnlineBackup"].astype(str) + "_" + X["DeviceProtection"].astype(str)
        X["PaperlessBilling_PaymentMethod"] = X["PaperlessBilling"].astype(str) + "_" + X["PaymentMethod"].astype(str)
        
        # Drop the original columns that were combined
        X = X.drop(self.col_to_drop_, axis=1)

        return X

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

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
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
    

    

    """
    Future functions to remember once we develop our skill:
    predict_proba: For models that can output probabilities, this function will return the probability estimates for each class.
    save: To save the trained model to disk for later use.
    load: To load a saved model from disk.
    evaluate: To assess the model's performance using metrics like accuracy, F1 score, etc.
    """

class RFCModel(BaseModel):
    def __init__(self, numerical_cols, categorical_cols) -> None:

        #1: Define feature engineering
        feature_engineering = FeaturePipeline()

        #2 preprocessing already done in FeaturePipeline

        #3 and #4: Model and pipeline

        my_pipeline = Pipeline(steps=[
            ("feature_engineering", feature_engineering),
            ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=7, max_depth=10, random_state=42))
        ])

        super().__init__(my_pipeline)

class KNNModel(BaseModel):
    def __init__(self, numerical_cols, categorical_cols) -> None:
        # Same as RFC but now with KNN model
        feature_engineering = FeaturePipeline()

        # 2 preprocessing already done in FeaturePipeline

        #KNN attempt try different feature scaling
        my_pipeline = Pipeline(steps=[
            ("feature_engineering", feature_engineering),
            ("model", KNeighborsClassifier(n_neighbors=5, weights = "distance", metric = "manhattan"))
        ])
        super().__init__(my_pipeline)

class LRModel(BaseModel):
    def __init__(self, numerical_cols, categorical_cols) -> None:
        # Same as RFC but now with Logistic Regression model

        feature_engineering = FeaturePipeline()

        #   preprocessing already done in FeaturePipeline

        my_pipeline = Pipeline(steps=[
            ("feature_engineering", feature_engineering),
            ("model", LogisticRegression(max_iter=1000))
        ])

        super().__init__(my_pipeline)


def plot_mi_scores(mi_scores: pd.Series) -> None:
    plt.figure(dpi=100, figsize=(12, 8))
    sns.barplot(x=mi_scores.values, y=mi_scores.index)
    plt.title("Mutual Information Scores")
    plt.xlabel("MI Score")
    plt.ylabel("Features")
    plt.show()

def make_mi_scores(X: pd.DataFrame, y: pd.Series, discrete_cols: pd.Series) -> pd.Series:
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_cols)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def get_score(model: object, X_valid: pd.DataFrame, y_valid: pd.Series) -> float:
    preds = model.predict(X_valid)
    return (f1_score(y_valid, preds), accuracy_score(y_valid, preds))

def get_cv_score(model: object, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    f1_cv_score = cross_val_score(model.pipeline, X_train, y_train, cv=5, scoring="f1").mean()
    acc_cv_score = cross_val_score(model.pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()
    return (f1_cv_score, acc_cv_score)

def can_be_number(s: pd.Series, threshhold = 0.9) -> bool:
    """
    Helper function to check if a pandas Series can be treated as numerical.
    It converts the Series to numeric, coercing errors to NaN, and then checks the proportion of non-NaN values.
    It returns True if the proportion of non-NaN values is greater than or equal to the specified threshhold.
    90% threshhold by default.
    """
    converted = pd.to_numeric(s, errors='coerce')
    return converted.notna().mean() >= threshhold
    
def get_user_data(X_train: pd.DataFrame, raw_feature_cols: list) -> pd.DataFrame:
    """
    Function to get user input for prediction.

    Problem: Tenure is registered as categorical data, so unfortunately the user has to input a "valid tenure" that already exists in the dataset.
    I could try to fix this by making tenure numerical. We'll see if I can do that in the future.
    So for now, I will leave it as is.

    FIXED: Now handles numerical inputs properly and gives feedback on invalid inputs.
    """

    user_data = {}
    numerical_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()


    for col in raw_feature_cols:
        # Show possible values for categorical columns
        if col not in numerical_cols and not can_be_number(X_train[col]):
            unique_vals = X_train[col].unique().tolist()
            print(f"\nPossible values for {col}: {unique_vals}")
        
        # while loop inside for loop to keep asking until valid input
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

if __name__ == "__main__":
    print("Loading Telco Customer Churn Data...")

    CustomerChurn = pd.read_csv("telco-churn-predictor\data\Telco-Customer-Churn.csv")

    # Customer ID is not useful for prediction; this is because the model starts to memorize the ID's instead of learning patterns.
    X = CustomerChurn.drop(["Churn", "customerID", "SeniorCitizen", "Partner", "Dependents"], axis=1)
    y = CustomerChurn.Churn.replace({"Yes": 1, "No": 0})

    raw_feature_cols = X.columns.tolist() # Used for user input later

    # Train size is 80% of data, random state is 42, stratify since it's a classification problem
    # Note: Stratify is useful when there is an imbalance in the target classes
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)


    # Problem FIXED: Corrected to use list comprehension to get column names based on dtype
    numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ["int64", "float64"]]
    categorical_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

    model_registry = {}
    model_full_name_registry = {
        "RFC": "Random Forest Classifier",
        "KNN": "K-Nearest Neighbors Classifier",
        "LR": "Logistic Regression Classifier"
    }
    mi_registry = {}
    cv_registry = {}


    print("Welcome to Customer Churn Prediction!\n")

    user = input("Please enter your name: ")

    # user = "Developer"  # For testing purposes


    if user == "Developer":
        print(f"Welcome back, {user}! You have developer access.\n")
        print("Menu Options:")
        print("1. See Mutual Information Scores")
        print("2. See MI Plot")
        print("3. Test Random Forest Classifier")
        print("4. Test K-Nearest Neighbors Classifier")
        print("5. Test Logistic Regression Classifier")
        print("6. Exit")

        while (c := input("Enter your choice (1-6): ")) != "6":
            match c:
                case "1" | "2":

                    print()
                    if "mi_scores" not in mi_registry:

                        feature_pipeline = FeaturePipeline()
                        X_train_fe = feature_pipeline.fit_transform(X_train)

                        discrete_features = [col in feature_pipeline.categorical_cols_ for col in X_train_fe.columns]

                        mi_scores = make_mi_scores(X_train_fe, y_train, discrete_features)

                        mi_registry["mi_scores"] = mi_scores
                        mi_registry["X_train_fe"] = X_train_fe
                        mi_registry["feature_pipeline"] = feature_pipeline

                    else:
                        mi_scores = mi_registry["mi_scores"]
                        X_train_fe = mi_registry["X_train_fe"]
                        feature_pipeline = mi_registry["feature_pipeline"]


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
                        model = model_class(numerical_cols, categorical_cols)
                        model.fit(X_train, y_train)
                        model_registry[model_name] = model
                    else:
                        model = model_registry[model_name]

                    score = get_score(model, X_valid, y_valid)

                    print(f"\n{model_full_name_registry[model_name]}")
                    print("------------------------------")
                    print("Evaluation on Validation Set:")
                    print(f"F1 Score: {score[0]:.4f}") # Print
                    print(f"Accuracy Score: {score[1]:.4f}\n") # Print Accuracy score with 4 decimal places (:.4f)

                    see_cv = input("Would you like to see Cross-Validation scores? (y/n)")
                    if see_cv.lower() == "y" and model_name not in cv_registry:
                        cv_score = get_cv_score(model, X_train, y_train)
                        cv_registry[model_name] = cv_score
                        print(f"Cross-Validation F1 Score: {cv_score[0]:.4f}")
                        print(f"Cross-Validation Accuracy Score: {cv_score[1]:.4f}\n")

                    elif see_cv.lower() == "y" and model_name in cv_registry:
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
            print("6. Exit\n")
            

    else:
        #Use user input to predict a churn or not with RFC only
        print(f"Welcome, {user}!\n")
        print("We'll be using the Random Forest Classifier for predictions.\n")
        
        if "RFC" not in model_registry:
            rfc_model = RFCModel(numerical_cols, categorical_cols)
            rfc_model.fit(X_train, y_train)
            model_registry["RFC"] = rfc_model
        else:
            rfc_model = model_registry["RFC"]

        while True:
            print("Please provide the following customer information for churn prediction:")

            userData = get_user_data(X_train, raw_feature_cols)

            prediction = rfc_model.predict(userData)

            if prediction[0] == 1:
                print("\nPrediction: The customer is likely to CHURN.")
            else:
                print("\nPrediction: The customer is likely to STAY.")


            print("Would you like to predict for another customer (y/n)?")
            if input().lower() != "y":
                break

        

    print("Exiting Customer Churn Prediction. Goodbye!")




























# Old code below for reference (super messy and flawed due to leakage issues, but kept for notes and observations)



# ---------------------------------------------------------------------------------------------------- OLD VERSION (FLAWED) ----------------------------------------------------------------------------------------------------
"""
--------------------------------- Notes & Observations ---------------------------------

Here are issues that GPT noticed in the current implementation along with suggestions for improvement:

1. Major leakage issue:
   - Here is the proposed solution from GPT to fix leakage (will have to rewrite significant portions of the code, so might as well reset):
   - Learn classes for Feature Engineering and Model (BaseModel and child classes).
    - Combine feature engineering, preprocessing, and model into a single Pipeline.

2. customerID:
   - High MI due to factorization. Model memorizes IDs and uses them to predict churn.
   - Action: Already dropped customerID from features.

3. Factorization of categorical features:
   - Current manual factorization can cause unseen categories in validation to fail.
   - Recommendation: Use OneHotEncoder or OrdinalEncoder within the pipeline for safety.

4. Mutual Information filtering:
   - Dropping features with MI < 0.02 slightly decreased CV F1, but caused large drop in validation F1.
   - Action: Reassess MI thresholds after applying proper pipeline preprocessing.

5. Feature combinations:
   - Some combined features improved MI significantly (e.g., Contract_Tenure, PaperlessBilling_PaymentMethod).
   - Keep meaningful combinations; discard marginal improvements to reduce dimensionality.

6. Pipeline structure:
   - Current preprocessing is outside model test functions, risking inconsistencies.
   - Action: Combine feature engineering, preprocessing, and model into a single pipeline.

7. Cross-validation vs. validation discrepancy:
   - CV F1 reasonable, but validation F1 drops drastically.
   - Likely caused by leakage, factorization, or preprocessing differences.

8. Scaling & preprocessing:
   - KNN is sensitive to feature scaling; ensure scaling happens after feature engineering.
   - RF less sensitive, but consistent preprocessing is best practice.

9. Code maintainability:
   - Repeated testing functions can be generalized to reduce duplication.
   - Document why columns are dropped or combined for clarity.

10. Next steps / TODO:
   - Rewrite program with proper pipeline structure to fix leakage.
   - Test meaningful feature combinations.
   - Keep notes of MI observations and cross-validation results.
   - Apply consistent preprocessing to train and validation sets.
   - Generalize model testing function to accept any classifier.

"""

# def combineCols(X_train : pd.DataFrame) -> pd.DataFrame:
    
#     X_train["tenure_bin"] = pd.cut(X_train.tenure, bins=[0, 6, 12, 24, 48, 75], labels=False, include_lowest=True)
#     X_train["Contract_Tenure"] = X_train.Contract.astype(str) + "_" + X_train.tenure_bin.astype(str)

    
#     X_train["OnlineSecurity_TechSupport"] = X_train.OnlineSecurity.astype(str) + "_" + X_train.TechSupport.astype(str)

#     X_train["OnlineBackup_DeviceProtection"] = X_train.OnlineBackup.astype(str) + "_" + X_train.DeviceProtection.astype(str)

#     X_train["PaperlessBilling_PaymentMethod"] = X_train.PaperlessBilling.astype(str) + "_" + X_train.PaymentMethod.astype(str)
    
#     # Drop the original columns that were combined
#     X_train = X_train.drop(["Contract","tenure", "tenure_bin", 
#                             "OnlineSecurity", "TechSupport", 
#                             "OnlineBackup", "DeviceProtection", 
#                             "PaperlessBilling", "PaymentMethod", 
#                             "gender", "PhoneService", "MultipleLines"],  # Removing columns with low MI scores
#                             axis=1)

# # BIGGEST ISSUE HERE: Factorization causes leakage AND corruption of data: FIX -> Use Ordinal Encoder
#     # Gets an array of all column names with object datatype, gets each column, and then factorizes it
#     for col in X_train.select_dtypes(include=["object"]).columns:
#         X_train[col], _ = X_train[col].factorize()

#     return X_train

# def plot_mi_scores(mi_scores: pd.Series) -> None:
#     plt.figure(dpi=100, figsize=(12, 8))
#     sns.barplot(x=mi_scores.values, y=mi_scores.index)
#     plt.title("Mutual Information Scores")
#     plt.xlabel("MI Score")
#     plt.ylabel("Features")
#     plt.show()

# def get_score(model: object, X_train: pd.DataFrame, X_valid: pd.DataFrame, y_train: pd.Series, y_valid: pd.Series) -> float:
#     preds = model.predict(X_valid)
#     return (f1_score(y_valid, preds), accuracy_score(y_valid, preds))

# def make_mi_scores(X: pd.DataFrame, y: pd.Series, discrete_cols: pd.Series) -> pd.Series:
#     mi_scores = mutual_info_classif(X, y, discrete_features=discrete_cols)
#     mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
#     mi_scores = mi_scores.sort_values(ascending=False)
#     return mi_scores

# def test_RFC(X_train: pd.DataFrame, preprocessor: Pipeline, X_valid: pd.DataFrame, y_train: pd.Series, y_valid: pd.Series) -> None:
#     # model = RandomForestClassifier(n_estimators=100, random_state=42)

#     my_pipeline = Pipeline(steps=[
#         ("preprocessor", preprocessor),
#         ("model", RandomForestClassifier(n_estimators=100, random_state=42))
#     ])
#     my_pipeline.fit(X_train, y_train)

#     score = get_score(my_pipeline, X_train, X_valid, y_train, y_valid)

#     print("Random Forest Classifier")
#     print("------------------------------")

#     print("Performing 5-Fold Cross-Validation...")
#     cv_score = cross_val_score(my_pipeline, X_train, y_train, cv=5, scoring="f1").mean()
#     print(f"Cross-Validation F1 Score: {cv_score:.4f}")
#     cv_score_acc = cross_val_score(my_pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()
#     print(f"Cross-Validation Accuracy Score: {cv_score_acc:.4f}")
#     print("------------------------------")

#     print("Evaluation on Validation Set:")
#     print(f"F1 Score: {score[0]:.4f}") # Print F1 score with 4 decimal places (:.4f)
#     print(f"Accuracy Score: {score[1]:.4f}") # Print Accuracy score with 4 decimal places (:.4f)

# def test_KNN(X_train: pd.DataFrame, preprocessor: Pipeline, X_valid: pd.DataFrame, y_train: pd.Series, y_valid: pd.Series) -> None:
    
#     my_pipeline = Pipeline(steps=[
#         ("preprocessor", preprocessor),
#         ("model", KNeighborsClassifier(n_neighbors=5))
#     ])

#     my_pipeline.fit(X_train, y_train)

#     score = get_score(my_pipeline, X_train, X_valid, y_train, y_valid)
#     print("K-Nearest Neighbors Classifier")
#     print("------------------------------")

#     print("Performing 5-Fold Cross-Validation...")
#     cv_score = cross_val_score(my_pipeline, X_train, y_train, cv=5, scoring="f1").mean()
#     print(f"Cross-Validation F1 Score: {cv_score:.4f}")
#     cv_score_acc = cross_val_score(my_pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()
#     print(f"Cross-Validation Accuracy Score: {cv_score_acc:.4f}")
#     print("------------------------------")

#     print("Evaluation on Validation Set:")
#     print(f"F1 Score: {score[0]:.4f}") # Print F1 score with 4 decimal places (:.4f)
#     print(f"Accuracy Score: {score[1]:.4f}") # Print Accuracy score with 4 decimal places (:.4f)
    
# def test_LR(X_train: pd.DataFrame, preprocessor: Pipeline, X_valid: pd.DataFrame, y_train: pd.Series, y_valid: pd.Series) -> None:
    
#     my_pipeline = Pipeline(steps=[
#         ("preprocessor", preprocessor),
#         ("model", LogisticRegression())
#     ])

#     my_pipeline.fit(X_train, y_train)

#     score = get_score(my_pipeline, X_train, X_valid, y_train, y_valid)
#     print("Logistic Regression Classifier")
#     print("------------------------------")

#     print("Performing 5-Fold Cross-Validation...")
#     cv_score = cross_val_score(my_pipeline, X_train, y_train, cv=5, scoring="f1").mean()
#     print(f"Cross-Validation F1 Score: {cv_score:.4f}")
#     cv_score_acc = cross_val_score(my_pipeline, X_train, y_train, cv=5, scoring="accuracy").mean()
#     print(f"Cross-Validation Accuracy Score: {cv_score_acc:.4f}")
#     print("------------------------------")

#     print("Evaluation on Validation Set:")
#     print(f"F1 Score: {score[0]:.4f}") # Print F1 score with 4 decimal places (:.4f)
#     print(f"Accuracy Score: {score[1]:.4f}") # Print Accuracy score with 4 decimal places (:.4f)


# if __name__ == "__main__":

#     print("Loading Customer Churn Data...")
#     CustomerChurn = pd.read_csv("WeekThree\\CustomerChurn\\WA_Fn-UseC_-Telco-Customer-Churn.csv")

#     # Customer ID is not useful for prediction; this is because the model starts to memorize the ID's instead of learning patterns.
#     X = CustomerChurn.drop(["Churn", "customerID", "SeniorCitizen", "Partner", "Dependents"], axis=1)
#     y = CustomerChurn.Churn.replace({"Yes": 1, "No": 0})

#     # Train size is 80% of data, random state is 42, stratify since it's a classification problem
#     # Note: Stratify is useful when there is an imbalance in the target classes
#     X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)

#     X_train = combineCols(X_train)
#     X_valid = combineCols(X_valid)

#     # # Identify categorical and numerical columns (Don't need categorical since all categories were transformed to numerical via factorization in combineCols)
#     numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ["int64", "float64"]]

#     numerical_Transformer = Pipeline(steps=[
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler())
#     ])
#     preprocessor = ColumnTransformer(transformers=[
#         ("num", numerical_Transformer, numerical_cols)
#     ])


#     print("Welcome to Customer Churn Prediction!\n")

#     print("Menu Options:")
#     print("1. See Mutual Information Scores")
#     print("2. See MI Plot")
#     print("3. Test Random Forest Classifier")
#     print("4. Test K-Nearest Neighbors Classifier")
#     print("5. Test Logistic Regression Classifier")
#     print("6. Exit")

#     while (c := input("Enter your choice (1-6): ")) != "6":
#         match c:
#             case "1":
#                 discrete_features = X_train.dtypes == "int64"
#                 mi_scores = make_mi_scores(X_train, y_train, discrete_features)
#                 print("Mutual Information Scores:")
#                 print(mi_scores)
            
#             case "2":
#                 discrete_features = X_train.dtypes == "int64"
#                 mi_scores = make_mi_scores(X_train, y_train, discrete_features)
#                 plot_mi_scores(mi_scores)

#             case "3":
#                 test_RFC(X_train, preprocessor, X_valid, y_train, y_valid)
            
#             case "4":
#                 test_KNN(X_train, preprocessor, X_valid, y_train, y_valid)
            
#             case "5":
#                 test_LR(X_train, preprocessor, X_valid, y_train, y_valid)

#             case _:
#                 print("Invalid choice. Please enter a number between 1 and 6.")
#                 continue
            
#         print("\nMenu Options:")
#         print("1. See Mutual Information Scores")
#         print("2. See MI Plot")
#         print("3. Test Random Forest Classifier")
#         print("4. Test K-Nearest Neighbors Classifier")
#         print("5. Test Logistic Regression Classifier")
#         print("6. Exit")


#     print("Exiting Customer Churn Prediction. Goodbye!")




