#No AI was used to generate this code. All logic authored by HG on 11/23/25
#imports
import sys
import os

#code is failing so adding this section to force the project root in the pythonpath

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

#remaining imports
import yaml
import numpy as np
import pandas as pd

from codes.load_data import load_raw_data

#pulling dataset info for initial inspection. Need to determine which fields I'll be using or if I'll need to manually derive fields from existing columns

raw_preview = load_raw_data()
print("Initial Data Preview:")
print(raw_preview.head())
print("\nColumn Info:")
print(raw_preview.info())
print("complete")


#manually deriving columns needed for this analysis
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    #doing Clean Up here


    #The duration column shows how long a phone call lasted with each customer but this is messy data because it only gets captured after a call ends with the customer
    #Duration is forward looking info and outside the scope of real time prediction making so will drop this category to keep the dataset clean and optimized for the case use

    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    #converting yes and no columns to 0/1 numeric 
    yes_no_cols = ["default", "housing", "loan"]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = (df[col] == "yes").astype(int)
    #y is term deposit subscription indicator, it labels yes/no if customer is subscribed or not. we don't need that field, we're only looking at surplus next month which derives from balance, so will drop this other unnecessary field

    if "y" in df.columns:
        df = df.drop(columns=["y"])    
	

    # Grounded financial behavior features


    #Step 1: Stability score based on balance and loan status
	# I'm adding an offset to balance before taking log to negative balances don't break the calculation
    balance_offset = 2000
    df["balance_log"] = np.log(df["balance"] + balance_offset)

    df["stability_score"] = (
        df["balance_log"]
        - 0.5 * df["housing"]	# having a housing loan reduces stability compared to not having a mortgage
        - 1.0 * df["loan"]	# having a personal load reduces stability even more than a housing loan 
    )

    #Step 2: Engagement behavior based on contact history
    df["campaign_intensity"] = df["campaign"] / (df["previous"] + 1)
    df["recent_contact"] = ((df["pdays"] >= 0) & (df["pdays"] < 30)).astype(int)    #pdays = days since lasy contact, recent is anything within 30 days 
    df["engagement_score"] = df["campaign_intensity"] + 0.5 * df["recent_contact"]

    #Step 3: Risk flag: negative balance
	# If balance is negative, flag = 1, when balance goes negative flag = 0
    df["risk_flag"] = (df["balance"] < 0).astype(int)

    #Step 4: Target variable for this project: surplus vs deficit next month given current day balance. Surplus next month = 1 if balance >= 0 , if balance is negative then 0
    df["surplus_next_month"] = (df["balance"] >= 0).astype(int)

    #One hot encode categorical variables
    categorical_cols = [
        "job",
        "marital",
        "education",
        "contact",
        "month",
        "poutcome",
    ]
    existing_cats = [c for c in categorical_cols if c in df.columns]
    df = pd.get_dummies(df, columns=existing_cats, drop_first=True)

    return df


def main():
    with open("configs/config.yaml") as f:	#load config
        config = yaml.safe_load(f)

    processed_path = config["data"]["processed_path"]
    file_name = config["data"]["monthly_features_file"]
    output_path = os.path.join(processed_path, file_name)

    raw_df = load_raw_data()	#load raw data and engineer features
    processed_df = engineer_features(raw_df)

    os.makedirs(processed_path, exist_ok=True)		#save final processed dataset
    processed_df.to_csv(output_path, index=False)

    print("Processed data saved to:", output_path)
    print("Processed shape:", processed_df.shape)


if __name__ == "__main__":
    main()
