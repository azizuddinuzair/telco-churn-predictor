# Streamlit Apps

This directory contains two separate Streamlit applications and one combined app (tabs) for the Telco Customer Churn Prediction system.

## Apps

### Combined App (`app.py`)
**Port:** configurable (default 8502)  
**Purpose:** Single URL with two tabs (User + Developer)

**Run:**
```bash
streamlit run streamlit_apps/app.py --server.port 8502

```

## Docker Deployment

For production, you can choose either approach:
- **Two containers** (separate URLs):
- Developer App: `localhost:8501`
- User App: `localhost:8502`
- **Single container** (one URL): run `app.py` with tabs.

Both options are supported by the existing Dockerfile/compose setup.
