import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Define the paths to the preprocessed datasets
project_root = "C:/Users/Lenovo/OneDrive/Desktop/ml-firewall-project"
datasets = {
    "mitm_dataset": os.path.join(project_root, "data/processed/mitm_dataset_preprocessed.csv"),
    "malware_dataset": os.path.join(project_root, "data/processed/malware_dataset_preprocessed.csv"),
    "password_dataset": os.path.join(project_root, "data/processed/password_dataset_preprocessed.csv"),
}

# Define the target column for each dataset
target_columns = {
    "mitm_dataset": "Label",  # Replace with the actual target column name for this dataset
    "malware_dataset": "Class",  # Replace with the actual target column name for this dataset
    "password_dataset": "label",  # Replace with the actual target column name for this dataset
}

# Directory to save the trained models
model_dir = os.path.join(project_root, "backend/src/models")
os.makedirs(model_dir, exist_ok=True)

# Train and save a CatBoost model for each dataset
for name, path in datasets.items():
    print(f"Processing dataset: {name}")
    
    # Load the preprocessed dataset
    data = pd.read_csv(path)
    target_column = target_columns[name]
    
    # Split features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Identify categorical columns
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    print(f"Categorical columns for {name}: {categorical_columns}")
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a CatBoost model with categorical features specified
    model = CatBoostClassifier(verbose=0, random_state=42)
    model.fit(X_train, y_train, cat_features=categorical_columns)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name}: {accuracy:.2f}")
    
    # Save the trained model
    model_path = os.path.join(model_dir, f"{name}_catboost_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")