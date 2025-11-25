#No AI was used to generate this code. All syntax is authored by HG on 11/23/25

#initial imports
import os
import yaml
import pandas as pd
from google.cloud import storage


#loading the bank marketing dataset 

def load_raw_data() -> pd.DataFrame:
    #Loading config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    raw_path = config["data"]["raw_path"]
    file_path = os.path.join(raw_path, "bank-full.csv")

    #the UCI bank marketing data is semicolon delimited
    df = pd.read_csv(file_path, sep=";")
    return df


#downloading the processed dataset from Google cloud storage

def load_processed_data_from_gcs() -> pd.DataFrame:

    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    cloud_cfg = config["cloud"]
    bucket_name = cloud_cfg["bucket_name"]
    blob_name = cloud_cfg["processed_blob_name"]
    credentials_file = cloud_cfg["credentials_file"]

#creating GCS client using the service account credentials
    client = storage.Client.from_service_account_json(credentials_file)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

#downloading remp local file
    temp_path = os.path.join("data", "processed", "monthly_from_gcs.csv")
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    blob.download_to_filename(temp_path)

#loading into pandas
    df = pd.read_csv(temp_path)
    return df



def main():
    #testing local raw loader
    raw_df = load_raw_data()
    print("Local raw shape:", raw_df.shape)

    #testing GCS processed loader
    processed_df = load_processed_data_from_gcs()
    print("GCS processed shape:", processed_df.shape)
    print(processed_df.head())


if __name__ == "__main__":
    main()
