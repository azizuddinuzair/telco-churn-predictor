# Telco Customer Churn Prediction

## **Overview**

This project predicts whether a telecom customer will **churn** (i.e., stop using the service) based on their account and service features.  

The project implements:  

- **Feature engineering**: Feature engineering to create meaningful combined features that increased mutual information with the target and improved model predictability
- **Machine learning models**: Random Forest Classifier, K-Nearest Neighbors (KNN), and Logistic Regression  
- **Dual interfaces**: Both console and web-based (Streamlit) interfaces for evaluating models and making predictions
- **Mutual Information (MI) and Cross Validation analysis** to identify important features and model quality
- **Model persistence**: Save and load trained models for reuse

> *I applied the concepts learned in [LearningML_Winter2025](https://github.com/azizuddinuzair/LearningML_Winter2025) to a real-world customer churn prediction problem, reinforcing and solidifying my understanding of feature engineering and model evaluation. I then refactored the project to try to follow proper software engineering practices with clean architecture.*
---

## **Project Structure**

- **src/models/**: ML models, feature engineering pipeline, and saved model storage
- **streamlit_apps/**: Web interfaces (developer and user apps)
- **data/**: Dataset location
- **archive/**: Legacy console interfaces and original implementation
- **Churn_StreamlitAPP_Screenshots/**: Screenshots of the Streamlit applications
- **Dockerfile & compose.yaml**: Docker configuration for containerized deployment (if ever needed)

---

## **Dataset**

- **Source:** `Telco-Customer-Churn.csv` sourced from [Kaggle]("https://www.kaggle.com/datasets/blastchar/telco-customer-churn") 
- **Target variable:** `Churn` (`Yes` → 1, `No` → 0)  
- **Preprocessing:** Dropped columns shown (via MI analysis or domain reasoning) to have little predictive value or to cause leakage: `customerID`, `gender`, `PhoneService`, `MultipleLines`, `SeniorCitizen`, `Partner`, `Dependents` 

---

## **Feature Engineering**

New features I created to capture interactions between existing columns:  

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
***Deployed*** -> Visit at: [Telco Customer Churn Predictor](https://telco-churn-predictor-9f3gy7wbrwmm5jsgarwfuq.streamlit.app/)

Or if you download on your own machine:
```bash
streamlit run streamlit_apps/app.py --server.port 8502
```

The Streamlit app provides a modern, interactive web interface with sidebar navigation:
- **User**: Make predictions on customer churn
- **Developer Tools**: MI Analysis, Model Testing, Model Comparison

4. Have Fun :)


<br>
<br>

## **Docker Deployment**

**Recommended** - Run via Docker Compose (single app with sidebar):
```bash
docker compose build
docker compose up
```
Access at `http://localhost:8502` with sidebar navigation for User and Developer Tools.

---

## **Refactoring Journey**

After completing the initial implementation, I refactored the entire project to follow proper software engineering practices. This included:

- **Separation of concerns**: Split the monolithic script into modular components (models, interfaces, utilities)
- **Feature pipeline isolation**: Moved `FeaturePipeline` into its own module to prevent code duplication
- **Model persistence**: Added save/load functionality so trained models can be reused without retraining
- **Dual interfaces**: Created both console and Streamlit web interfaces to serve different use cases
- **Clean architecture**: Organized code into logical packages (models, interfaces, utils) for better maintainability


The original code is preserved in `src/old_telco_churn_predictor.py` for reference, and is also available at [LearningML_Winter2025](https://github.com/azizuddinuzair/LearningML_Winter2025)
