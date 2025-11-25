# Behavioral Cashflow Reliability Tracker  
### *End to End Machine Learning Pipeline with Cloud Data, Experiment Tracking, Docker Deployment, API Serving, and Streamlit Front End*

---

## Overview

This project implements a complete **End to end MLOps workflow** for a real financial prediction task:  
**Predicting whether a customer will have either a positive (surplus) or negative(deficit) cashflow next month based on behavioral banking data**

**no AI was used to generate these codes. All syntax is authored by Hadil Ghazal (Nov 2025).

- Cloud hosted dataset via GCS  
- Fully reproducible ML pipeline  
- MLflow experiment tracking  
- Docker container  
- Production API deployed to the cloud run  
- Streamlit front end calling the API  
- Configuration driven model training  
- Clean README with reproducibility instructions and background info as well 

**Note:** Model accuracy is *not the objective* ,the goal is to deploy a full and complete ML system

---

#  Live Deployment Link

### Streamlit App here:  
https://behavioral-model-reliability-tracker-wxuzabeawnyyxg2gxr6uf4.streamlit.app/


---

# Project background and wWhy This Matters

Banks analyze customer cashflow patterns to understand:

- Overdraft risk  
- Loan repayment stability  
- Income vs spending behavior  
- Early signs of financial stress  

Traditional systems react **after** problems appear  
This project explores a proactive alternative:

### Predict next month cashflow direction (either surplus or deficit)

This is a simple but powerful behavioral signal:  
It is easy to compute, explainable, reproducible, and has a significant capacity to be expanded on and later developed and implemented in banking contexts

---

# Repository Structure

```
behavioral-model-reliability-tracker:
1. codes contains:
- load_data.py
- preprocessing.py
- train.py
- api.py
2. configs/ contains:
- config.yaml
3. models
4. data
5. app.py (streamlit front end app)
6. Dockerfile (API container)
7. requirements.txt
8. README.md
9. gcp-service-account.json(Gitignored)



---

# Dataset Description

**Dataset:** Bank Marketing Dataset (UCI ML Repository)  
Link: https://archive.ics.uci.edu/dataset/222/bank+marketing

### Feature Engineering  
We map the dataset into behavioral banking signals:

- balance  
- housing loan  
- personal loan  
- contact frequency  
- previous month outcomes  
- month of the year (seasonality)  
- proxy for “overdrawn behavior” using pdays/previous  

### Target  
`surplus_next_month = 1 if next_month_balance > this_month_balance else 0`

This transforms the original dataset into a **binary classification forecast**.

---

# Model Description

We intentionally use a **simple Logistic Regression model** because:

- Fully deterministic  
- Highly reproducible  
- Interpretable  
- Works for behavioral tabular data  
- Perfect for pipelines and MLOps assignments  

All configuration values (split ratios, random seed, model type) are stored in:

```
configs/config.yaml
```

---

# Google Cloud Data Storag

Processed features are uploaded to:

```
gs://behavioral-ml-reliability-data/processed/monthly_cashflow_features.csv
```

`load_data.py` loads data programmatically using a service account JSON key

---

# Experiment Tracking(MLflow)

Each training run logs:

- train/validation/test accuracy  
- runtime parameters  
- dataset splits  
- trained model artifact  
- feature column list  

All runs stored under mlruns

---

# Docker Containerization

A complete Dockerfile is included:

- Based on python:3.10  
- Installs dependencies  
- Copies API code  
- Launches Uvicorn server  

### testing locally:

```bash
docker build -t behavioral-ml-reliability-api .
docker run -p 8080:8080 behavioral-ml-reliability-api
```

---

# Cloud Deployment(Google Cloud Run)

The API service container is deployed using:

- Google Artifact Registry  
- Google Cloud Run  
- HTTPS public endpoint  
-  IAM disabled for public access testing  

The deployed API responds to:
GET / to service info
POST /predict to model prediction

---

# Streamlit Front end

The front end UI:

- Accepts user behavior inputs  
- Calls the Cloud Run API via HTTPS  
- Displays surplus/deficit prediction  
- Provides user friendly explanation  

Local launch:

```bash
streamlit run app.py
```

---

# HOW TO: Reproducibility step by step

### Step 1. Clone repository

```bash
git clone https://github.com/hadil-ghazal/behavioral-model-reliability-tracker.git
cd behavioral-model-reliability-tracker
```

### Step 2. Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3. Preprocess dataset

```bash
python codes/preprocessing.py
```

### Step 4. Train model + log experiments

```bash
python codes/train.py
```

### Step 5. Run API locally

```bash
uvicorn codes.api:app --reload
```

### Step 6. Build Docker image

```bash
docker build -t behavioral-ml-reliability-api .
```

### Step 7. Deploy to Cloud Run  
(*performed manually via UI or gcloud tools*)

### Step 8. Launch Streamlit frontend

```bash
streamlit run app.py
```

---

# Step by Step Checklist - COMPLETED RUBRIC REQUIREMENTS

## 1. Choose a Dataset/ Define Problem
- Used UCI Bank Marketing dataset  
- Engineered cashflow features  
- Defined classification target (surplus vs deficit)  
- Metric = accuracy  

##  2. Cloud Data Storage
- Processed dataset uploaded to GCS  
- Programmatic access implemented  
- Load tested successfully  

##  3. Reproducible ML Pipeline
- 1)Data reading 2)preprocessing 3) split 4) train 5)evaluate  
- Config driven architecture  
- Deterministic splits  

##  4. Experiment Tracking
- MLflow run logging  
- Model artifacts stored  
- Version controlled experiments  

##  5. Docker Container
- Reproducible environment  
- API packaged  
- Local container tested  

##  6. Model API
- FastAPI backend  
- predict endpoint  
- JSON schema validation  
- Local and cloud tested  

##  7. Cloud Deployment
- Built and pushed container  
- Deployed to Cloud Run  
- Public endpoint verified  

## 8. Front End Interface
- Streamlit UI  
- Sends requests to Cloud Run  
- Displays predictions  
- Fully functional demo  

## 9. Documentation
- This README  
- Clear runnable instructions  
- Links to deployed systems  
- Architecture and narrative  

---

# Takeaways

This project demonstrates that:

- A small, interpretable model can power a full production style system  
- Modern MLOps tools enable one person to build an entire ML workflow  
- Reliability, tracking, and reproducibility matter far more than accuracy  
- Cloud hosted APIs and simple UIs lead to real world usability  





