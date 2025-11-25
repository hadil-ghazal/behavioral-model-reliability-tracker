#No AI was used to generate this code. All syntax is authored by HG on 11/23/25

#initial imports
import os
import yaml
import pandas as pd

#loading the bank marketing dataset 

def load_raw_data() -> pd.DataFrame:
    #Loading config
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    raw_path = config["data"]["raw_path"]
    file_name = "bank-full.csv"
    file_path = os.path.join(raw_path, file_name)

    #the UCI bank marketing data is semicolon delimited
    df = pd.read_csv(file_path, sep=";")
    return df

def main():
    df = load_raw_data()
    print("Loaded shape:", df.shape)
    print(df.head())

if __name__ == "__main__":
    main()


