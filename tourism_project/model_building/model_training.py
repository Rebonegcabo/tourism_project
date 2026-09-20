
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from huggingface_hub import HfApi, login

# Hugging Face login (token will be passed via environment variable in GitHub Actions)
try:
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if HF_TOKEN:
        login(token=HF_TOKEN)
    else:
        print("HF_TOKEN environment variable not set. Model push to Hugging Face may fail.")
except Exception as e:
    print(f"Could not log in to Hugging Face: {e}")

# Define paths (relative to the repository root when run in GitHub Actions)
data_path = 'tourism_project/data/tourism.csv'
encoder_path = 'tourism_project/deployment/encoder.joblib' # Path to the saved encoder from data-prep
model_output_path = 'tourism_project/deployment/rf_model.joblib'

# Ensure deployment directory exists for model and encoder
os.makedirs(os.path.dirname(model_output_path), exist_ok=True)

# Load the dataset
df = pd.read_csv(data_path)

# Load the pre-fitted OneHotEncoder
encoder = joblib.load(encoder_path)

# Identify categorical columns (must be consistent with data_preparation.py)
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# Apply one-hot encoding to the full dataset for training
encoded_features = encoder.transform(df[categorical_cols])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)

df_encoded = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

# Define features (X) and target (y)
# Drop 'ProdTaken' (target), 'Unnamed: 0' (artifact), and 'CustomerID' (identifier) from features
X = df_encoded.drop(['ProdTaken', 'Unnamed: 0', 'CustomerID'], axis=1, errors='ignore') # 'errors' to handle if they were already dropped
y = df_encoded['ProdTaken']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize and train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
rf_y_pred = rf_model.predict(X_test)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_test, rf_y_pred)
rf_report = classification_report(y_test, rf_y_pred)

print(f"Random Forest Model Accuracy: {rf_accuracy:.4f}")
print("Random Forest Classification Report:
", rf_report)

# Save the trained Random Forest model
joblib.dump(rf_model, model_output_path)
print(f"Random Forest model saved to {model_output_path}")

# --- Push model and encoder to Hugging Face Model Hub ---

HF_MODEL_REPO_ID = "DrGee/tourism-package-model" # Replaced 'Rebonegcabo' with 'DrGee'
HF_DATASET_REPO_ID = "DrGee/tourism-package-prediction" # Replaced 'Rebonegcabo' with 'DrGee'

api = HfApi()

# Create model repo if it doesn't exist
try:
    api.create_repo(repo_id=HF_MODEL_REPO_ID, repo_type="model", private=False, token=HF_TOKEN)
    print(f"Created Hugging Face model repository: {HF_MODEL_REPO_ID}")
except Exception as e:
    print(f"Could not create Hugging Face model repository (might already exist): {e}")

# Upload model
try:
    api.upload_file(
        path_or_fileobj=model_output_path,
        path_in_repo="rf_model.joblib",
        repo_id=HF_MODEL_REPO_ID,
        repo_type="model",
        token=HF_TOKEN,
    )
    print(f"Uploaded rf_model.joblib to {HF_MODEL_REPO_ID}")
except Exception as e:
    print(f"Could not upload rf_model.joblib: {e}")

# Upload encoder to the dataset repo (as it's used for preprocessing the data for the model)
try:
    api.upload_file(
        path_or_fileobj=encoder_path,
        path_in_repo="encoder.joblib",
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset", # Upload to dataset repo as it's data-related
        token=HF_TOKEN,
    )
    print(f"Uploaded encoder.joblib to {HF_DATASET_REPO_ID}")
except Exception as e:
    print(f"Could not upload encoder.joblib: {e}")
