# Streamlit Apps

This directory contains two separate Streamlit applications and one combined app (tabs) for the Telco Customer Churn Prediction system.

## Apps

### Web Interface (`app.py`)
***Deployed*** -> Visit at: [Telco Customer Churn Predictor](https://telco-churn-predictor-9f3gy7wbrwmm5jsgarwfuq.streamlit.app/)


```

## Docker Deployment

**Recommended** - Run via Docker Compose:
```bash
cd ..
docker compose build
docker compose up
```
Access at `http://localhost:8502` with sidebar navigation.

**Alternative** - Run locally without Docker:
```bash
streamlit run streamlit_apps/app.py --server.port 8502
```
