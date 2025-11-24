#No AI was used to generate this content authored by HG on 11/21/25
#initial setup to direct pipeline where data is and how model should be trained
data:
  raw_path: "data/raw"
  processed_path: "data/processed"
  monthly_features_file: "monthly_cashflow_features.csv"

model:
  type: "logistic_regression"
  target: "surplus_next_month"
  test_size: 0.2 #im choosing default split in scikit-learn 
  val_size: 0.1
  random_state: 42
