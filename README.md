# Telco Customer Churn Prediction

## **Overview**

This project predicts whether a telecom customer will **churn** (i.e., stop using the service) based on their account and service features.  

Key Highlights:  

- **Feature engineering**: Created interaction features improving mutual information with the target
- **Machine learning models**: Random Forest (CV F1=0.569, Accuracy=0.796) outperforms baseline Logistic Regression and KNN, with full CV evaluation.
- **Engineering & interfaces**: Streamlit web interface for developer experimentation and user predictions, built on modular, maintainable architecture.
- **Deployment-ready**: Dockerized with optional Compose setup

---

## **Project Structure**

- **src/models/**: ML models, feature engineering pipeline, and saved model storage
- **streamlit_apps/**: Web interfaces (developer and user apps)
- **data/**: Dataset location
- **archive/**: Old console interfaces and original implementation
- **Churn_StreamlitAPP_Screenshots/**: Screenshots of the Streamlit applications
- **Dockerfile & compose.yaml**: Docker configuration for containerized deployment (if ever needed)

---

## **Dataset**

- **Source:** `Telco-Customer-Churn.csv` sourced from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 
- **Target variable:** `Churn` (`Yes` → 1, `No` → 0)  
- **Preprocessing:** Dropped columns shown (via MI analysis or domain reasoning) to have little predictive value or to cause leakage: `customerID`, `gender`, `PhoneService`, `MultipleLines`, `SeniorCitizen`, `Partner`, `Dependents` 

---

## **Feature Engineering**

New features I created to capture interactions between existing columns:  

- `tenure_bin`: Categorized tenure into bins to simplify dataset
- `Contract_Tenure`: Contract type x tenure bin
- `OnlineSecurity_TechSupport`: Online security x tech support   
- `OnlineBackup_DeviceProtection`: Online Backup x device protection services  
- `PaperlessBilling_PaymentMethod`: Billing method x payment method  

Columns with low predictive value or redundancy were removed. Features improved model predictability (measured via mutual information).

---

## **Models**

Three machine learning models were trained and evaluated:

| Model                              | Validation F1 / Accuracy | CV F1 / Accuracy | Notes                                                                        |
| ---------------------------------- | ------------------------ | ---------------- | ---------------------------------------------------------------------------- |
| **Random Forest Classifier (RFC)** | 0.570 / 0.792            | 0.569 / 0.796    | Best overall; tuned `n_estimators=200`, `max_depth=10`, `min_samples_leaf=7` |
| **K-Nearest Neighbors (KNN)**      | 0.511 / 0.749            | 0.542 / 0.765    | Shows slight overfitting; CV F1 below RFC                                    |
| **Logistic Regression (LR)**       | 0.557 / 0.784            | 0.586 / 0.799    | Baseline model; stable performance across validation and CV                        |


---

## **How to Run**

### **Installation**

1. Install required packages:
```bash
pip install -r requirements.txt
```

### **Web Interface (Streamlit)**
***Deployed*** -> Visit at: [Telco Customer Churn Predictor](https://telco-churn-predictor-9f3gy7wbrwmm5jsgarwfuq.streamlit.app/)

Or if you download on your own machine:
```bash
streamlit run streamlit_apps/app.py --server.port 8502
```

The Streamlit app provides an interactive web interface with sidebar navigation:
- **User**: Make predictions on customer churn
- **Developer Tools**: MI Analysis, Model Testing, Model Comparison

You can also view screenshots in [telco-churn-predictor/Churn_StreamlitAPP_Screenshots](https://github.com/azizuddinuzair/telco-churn-predictor/tree/main/Churn_StreamlitAPP_Screenshots)

<br>

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

After completing the initial implementation, I refactored the entire project to try follow proper software engineering practices. This included:

- **Modular Architecture and Code Organization**: Separated models, feature pipelines, and utilities into independent, reusable packages for maintainability and clarity.
- **Persistent Models**: Added save/load functionality so trained models can be reused without retraining
- **Web Interface**: Streamlit app for both developer tools and end-user predictions.


The original code is preserved in `archives/old_telco_churn_predictor.py` for reference, and is also available at [LearningML_Winter2025](https://github.com/azizuddinuzair/LearningML_Winter2025)
