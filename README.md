# Telco Customer Churn Prediction

## **Overview**

This project predicts whether a telecom customer will **churn** (i.e., stop using the service) based on their account and service features.  

The project implements:  

- **Feature engineering**: Feature engineering to create meaningful combined features that increased mutual information with the target and improved model predictability
- **Machine learning models**: Random Forest Classifier, K-Nearest Neighbors (KNN), and Logistic Regression  
- **Dual interfaces**: Both console and web-based (Streamlit) interfaces for evaluating models and making predictions
- **Mutual Information (MI) and Cross Validation analysis** to identify important features and model quality
- **Model persistence**: Save and load trained models for reuse

> *I applied the concepts learned in week 2 to a real-world customer churn prediction problem, reinforcing and solidifying my understanding of feature engineering and model evaluation. I then refactored the project to follow proper software engineering practices with clean architecture and separation of concerns.*
---

## **Project Structure**

The project has been refactored into a clean, modular architecture:

```
telco-churn-predictor/
├── src/
│   ├── models/              # Model classes and feature engineering
│   ├── interfaces/          # Console UI for developer and user modes
│   ├── data_loader.py       # Data loading utilities
│   └── utils.py             # Helper functions (MI scores, evaluation)
├── streamlit_apps/
│   ├── developer_app.py     # Web interface for developers
│   └── user_app.py          # Web interface for end users
├── data/                    # Dataset location
├── run_app.py              # Console entry point
└── requirements.txt        # Dependencies
```

---

## **Dataset**

- **Source:** `Telco-Customer-Churn.csv` sourced from [Kaggle]("https://www.kaggle.com/datasets/blastchar/telco-customer-churn") 
- **Target variable:** `Churn` (`Yes` → 1, `No` → 0)  
- **Preprocessing:** Dropped columns shown (via MI analysis or domain reasoning) to have little predictive value or to cause leakage: `customerID`, `gender`, `PhoneService`, `MultipleLines`, `SeniorCitizen`, `Partner`, `Dependents` 

---

## **Feature Engineering**

New features were created to capture interactions between existing columns:  

- `tenure_bin`: Categorized tenure into bins so that it simplifies the dataset and reveals relationships that would otherwise be hidden
- `Contract_Tenure`: Combination of contract type and tenure bin
- `OnlineSecurity_TechSupport`: Combination of online security and tech support subscription  
- `OnlineBackup_DeviceProtection`: Combination of backup and device protection services  
- `PaperlessBilling_PaymentMethod`: Combination of billing method and payment method  

Columns with low predictive value or redundancy were removed, such as `Contract`, `tenure`, and gender-related columns.  

---

## **Models**

Three machine learning models were trained and evaluated:

1. **Random Forest Classifier (RFC)**  
   - Tuned with `n_estimators=200`, `max_depth=10`, `min_samples_leaf=7`  
   - Best performing model overall  

2. **K-Nearest Neighbors (KNN)**  
   - Tuned with `weights="distance"` and `metric="manhattan"`  
   - Shows some overfitting; CV F1 score slightly lags validation F1  

3. **Logistic Regression (LR)**  
   - Baseline/control model used to judge the performance of the other models
   - Stable performance between validation and CV scores  

---

## **How to Run**

### **Installation**

1. Install required packages:
```bash
pip install -r requirements.txt
```

### **Console Version**

2. Run the console application:
```bash
python run_app.py
```

3. Enter "Developer" (case sensitive) to access developer mode with MI analysis and model comparison. Otherwise, enter your name for user mode to make predictions.

### **Web Interface (Streamlit)**

**Developer App** (Port 8501) - For model analysis and testing:
```bash
streamlit run streamlit_apps/developer_app.py --server.port 8501
```

**User App** (Port 8502) - For making predictions:
```bash
streamlit run streamlit_apps/user_app.py --server.port 8502
```

The Streamlit apps provide a modern, interactive web interface. The developer app is great for exploring feature importance and comparing models, while the user app offers a simple form-based interface for predictions.

4. Have Fun :)


<br>
<br>

## **Refactoring Journey**

After completing the initial implementation, I refactored the entire project to follow proper software engineering practices. This included:

- **Separation of concerns**: Split the monolithic script into modular components (models, interfaces, utilities)
- **Feature pipeline isolation**: Moved `FeaturePipeline` into its own module to prevent code duplication
- **Model persistence**: Added save/load functionality so trained models can be reused without retraining
- **Dual interfaces**: Created both console and Streamlit web interfaces to serve different use cases
- **Clean architecture**: Organized code into logical packages (models, interfaces, utils) for better maintainability

The original code is preserved in `src/old_telco_churn_predictor.py` for reference.

---

## Week Three Reflection:

### *Key Mistakes and Corrections*
- **Data Leakage through identifiers**<br>customerID initially had an extremely high MI score. This was due to the model learning ID patterns, so I removed the column entirely to prevent leakage.
- **Preprocessing outside pipelines**<br>Early versions of the code applied feature engineering and preprocessing outside the model pipeline, which caused data leakage and inconsistency. I resolved this by creating a `FeaturePipeline` class to handle all preprocessing within the pipeline.
- **Overfitting/Underfitting not visible through validation**<br>  KNN performed well on the validation set but failed under cross-validation, highlighting the importance of cross-validation for assessing model performance.
- **Mishandled a categorical column with High Cardinality**<br>`TotalCharges` was a categorical column that should've been numerical but wasn't because it was stored as a string. I converted it to numeric at the end to ensure proper modeling.
- **Poor code organization**<br>The original implementation had everything in one file, making it hard to maintain and test. I refactored it into a proper project structure with separate modules for models, interfaces, and utilities.

