
import pandas as pd
import joblib
import os
from sklearn.preprocessing import OneHotEncoder

# Define paths (these paths are relative to the repository root when run in GitHub Actions)
data_path = 'data/tourism.csv'
encoder_output_path = 'deployment/encoder.joblib'

# Ensure output directory exists for the encoder (relative to current script execution)
os.makedirs(os.path.dirname(encoder_output_path), exist_ok=True)

# Load the dataset
df = pd.read_csv(data_path)

# Identify categorical columns (excluding the target variable and CustomerID index)
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Apply one-hot encoding and save the encoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(df[categorical_cols])
joblib.dump(encoder, encoder_output_path)

print(f"OneHotEncoder saved to {encoder_output_path}")
