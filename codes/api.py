#No AI was used to generate this code. All code authored by Hadil Ghazal on 11/24/25

#imports
import os
import sys

#forcing python to read the project root since it crashes without this section of code

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#continuing imports 
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import yaml


#loading config model and feature columns 
with open(os.path.join(project_root, "configs", "config.yaml")) as f:
    CONFIG = yaml.safe_load(f)

MODEL_PATH = os.path.join(project_root, "models", "behavioral_model.joblib")
COLUMNS_PATH = os.path.join(project_root, "models", "feature_columns.joblib")

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(COLUMNS_PATH)



#defining the input schema for the API

class CashflowInput(BaseModel):
    # These are user-facing, raw-ish inputs. Internally we'll build engineered features.
    age: int
    job: Literal[
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown"
    ]
    marital: Literal["married", "single", "divorced"]
    education: Literal["primary", "secondary", "tertiary", "unknown"]
    default: Literal["yes", "no"]
    balance: int
    housing: Literal["yes", "no"]
    loan: Literal["yes", "no"]
    contact: Literal["cellular", "telephone"]
    month: Literal["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    campaign: int         #this is the number of contacts
    pdays: int            #this is days since last contact where -1 means never contacted
    previous: int         #previous is the number of times the customer was contacted in earlier marketing campaigns , defined as historical contacts, not calendar days
    poutcome: Literal["failure", "nonexistent", "success", "unknown"]

#helpers to engineer features in the same way as the preprocessing

def engineer_features_for_inference(input_data: CashflowInput) -> pd.DataFrame: #rebuilding a single row dataframe for the model

#creating dataframe that matches the original raw schema

    raw_dict = {
        "age": input_data.age,
        "job": input_data.job,
        "marital": input_data.marital,
        "education": input_data.education,
        "default": input_data.default,
        "balance": input_data.balance,
        "housing": input_data.housing,
        "loan": input_data.loan,
        "contact": input_data.contact,
        "day": 1,                 #not used in engineered features
        "month": input_data.month,
        "duration": 0,            #dropped in training
        "campaign": input_data.campaign,
        "pdays": input_data.pdays,
        "previous": input_data.previous,
        "poutcome": input_data.poutcome,
        "y": "no",                #placeholder bc original target not used for prediction
    }

    df = pd.DataFrame([raw_dict])


#cleaning data and converting formats to compatible forms

#converting the yes/no to 0/1

    for col in ["default", "housing", "loan"]:
        df[col] = (df[col] == "yes").astype(int)

# dropping term deposit subscription indicator to preserve surplus prediction scope
    df = df.drop(columns=["y"])


#grounded financial behavior defining

    # balance can be negative and offset before log to keep it defined
    balance_offset = 2000
    df["balance_log"] = np.log(df["balance"] + balance_offset)

    df["stability_score"] = (
        df["balance_log"]
        - 0.5 * df["housing"]      #housing loan increases obligations
        - 1.0 * df["loan"]         #personal loan increases obligations as well
    )

#engagement behavior based on contacts
    df["campaign_intensity"] = df["campaign"] / (df["previous"] + 1)

#previous days (pdays) is days since last outreach, considered recent if within 30 days

    df["recent_contact"] = ((df["pdays"] >= 0) & (df["pdays"] < 30)).astype(int)

    df["engagement_score"] = df["campaign_intensity"] + 0.5 * df["recent_contact"]

#risk flag to capture when balance goes negative
    df["risk_flag"] = (df["balance"] < 0).astype(int)

#One hot encode categorical variables

    categorical_cols = [
        "job",
        "marital",
        "education",
        "contact",
        "month",
        "poutcome",
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


#filling any missing values with 0
    df_aligned = df.reindex(columns=feature_columns, fill_value=0.0)

#cleaning nulls 

    df_aligned = df_aligned.replace([np.inf, -np.inf], np.nan)
    df_aligned = df_aligned.fillna(0.0)

    return df_aligned


#FastAPI app

app = FastAPI(title="Behavioral Cashflow Reliability Model API")


@app.get("/")
def read_root():
    return {
        "message": "Behavioral cashflow model API running",
        "description": "Predict surplus vs deficit next month based on cashflow behavior",
    }

#preducting whether next month is likely to be in a surplus or a deficit along with the model's predicted probability
@app.post("/predict")
def predict_surplus(input_data: CashflowInput):
    features = engineer_features_for_inference(input_data)
    proba = model.predict_proba(features)[0, 1]
    prediction = int(proba >= 0.5)

    return {
        "surplus_probability": float(proba),
        "surplus_prediction": prediction,
    }

