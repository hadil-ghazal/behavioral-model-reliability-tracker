#No AI was used to generate this code. All syntax authored by HG on 11/24/25

#Initial Imports

import os
import sys

#Python isn't automatically picking up the folder root path, have to hard code it so script doesnt fail
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#remaining imports now

import yaml
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from codes.load_data import load_processed_data_from_gcs


def load_config():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    return config


#splitting the processed dataset into train test and validation sets using the ratios setup and defined in config
def split_data(df, config):

    #doing cleanup so the model does not see inf or NaN values
    df = df.replace([np.inf, -np.inf], pd.NA)  # converting infinite values into NA
    df = df.dropna(axis=0)                    #code failing because of nulls, so dropping any row that still has NA
    df = df.astype(float)                     # standardizing all data into numeric

    target_col = config["model"]["target"]
    test_size = config["model"]["test_size"]
    val_size = config["model"]["val_size"]
    random_state = config["model"]["random_state"]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    #Split 1 : into training and temporary bucket
    total_test_val = test_size + val_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=total_test_val,
        random_state=random_state,
        stratify=y,
    )

    #Split 2: splitting temporary into validation and test
    val_ratio_of_temp = val_size / total_test_val

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_ratio_of_temp),
        random_state=random_state,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, X.columns.tolist()


#training a base regression model here
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


#calculating accuracy metrics on train validation and the test sets
def evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test):
    metrics = {}

    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        y_pred = model.predict(X_split)
        acc = accuracy_score(y_split, y_pred)
        metrics[f"{split_name}_accuracy"] = acc

    return metrics


#saving the trained model and column order to the models folder location
def save_model(model, feature_columns):
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "behavioral_model.joblib")
    columns_path = os.path.join("models", "feature_columns.joblib")

    joblib.dump(model, model_path)
    joblib.dump(feature_columns, columns_path)

    print(f"Saved model to: {model_path}")
    print(f"Saved to: {columns_path}")


def main():

    config = load_config()      #loading configuration
    df = load_processed_data_from_gcs() #loading processed data from GCS
    print("Loaded processed data shape from GCS:", df.shape)

    #Splitting into train/val/test
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_columns,
    ) = split_data(df, config)

    model = train_model(X_train, y_train)       #training model   

    #Evaluate the model
    metrics = evaluate_model(model, X_train, X_val, X_test, y_train, y_val, y_test)
    print("Model accuracy metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    save_model(model, feature_columns) #saving the model and feature columns


if __name__ == "__main__":
    main()

