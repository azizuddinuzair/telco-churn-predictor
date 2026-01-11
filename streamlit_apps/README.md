# Streamlit Apps

This directory contains two separate Streamlit applications for the Telco Customer Churn Prediction system.

## Apps

### 1. Developer App (`developer_app.py`)
**Port:** 8501  
**Purpose:** Advanced model analysis and testing

**Features:**
- Mutual Information Analysis with visualizations
- Individual model testing (RFC, KNN, Logistic Regression)
- Model comparison across all classifiers
- Cross-validation results
- Performance metrics (F1, Accuracy)

**Run:**
```bash
streamlit run streamlit_apps/developer_app.py --server.port 8501
```

### 2. User App (`user_app.py`)
**Port:** 8502  
**Purpose:** Customer churn predictions

**Features:**
- User-friendly input forms
- Real-time churn predictions
- Actionable recommendations
- Input validation
- Results visualization

**Run:**
```bash
streamlit run streamlit_apps/user_app.py --server.port 8502
```

## Docker Deployment

For production, both apps will run in separate containers:
- Developer App: `localhost:8501`
- User App: `localhost:8502`

This allows for independent scaling and different access controls per role.
