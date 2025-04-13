import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Define the paths to the preprocessed datasets and trained models
project_root = "C:/Users/Lenovo/OneDrive/Desktop/ml-firewall-project"
datasets = {
    "mitm_dataset": {
        "data_path": os.path.join(project_root, "data/processed/mitm_dataset_preprocessed.csv"),
        "catboost_model_path": os.path.join(project_root, "backend/src/models/mitm_dataset_catboost_model.pkl"),
        "xgboost_model_path": os.path.join(project_root, "backend/src/models/mitm_dataset_xgboost_model.pkl"),
        "target_column": "Label"
    },
    "malware_dataset": {
        "data_path": os.path.join(project_root, "data/processed/malware_dataset_preprocessed.csv"),
        "catboost_model_path": os.path.join(project_root, "backend/src/models/malware_dataset_catboost_model.pkl"),
        "xgboost_model_path": os.path.join(project_root, "backend/src/models/malware_dataset_xgboost_model.pkl"),
        "target_column": "Class"
    },
    "password_dataset": {
        "data_path": os.path.join(project_root, "data/processed/password_dataset_preprocessed.csv"),
        "catboost_model_path": os.path.join(project_root, "backend/src/models/password_dataset_catboost_model.pkl"),
        "xgboost_model_path": os.path.join(project_root, "backend/src/models/password_dataset_xgboost_model.pkl"),
        "target_column": "label"
    }
}

# Directory to save the Voting Classifier models
voting_model_dir = os.path.join(project_root, "backend/src/models")
os.makedirs(voting_model_dir, exist_ok=True)

# Process each dataset
for dataset_name, dataset_info in datasets.items():
    print(f"Processing dataset: {dataset_name}")
    
    # Load the preprocessed dataset
    data = pd.read_csv(dataset_info["data_path"])
    target_column = dataset_info["target_column"]
    
    # Split features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Encode the target column if it contains string labels
    if y.dtype == 'object':
        print(f"Encoding target column '{target_column}' for {dataset_name}...")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print(f"Classes for {dataset_name}: {list(label_encoder.classes_)}")
    
    # Identify categorical columns (if needed for CatBoost)
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_columns:
        print(f"Categorical columns for {dataset_name}: {categorical_columns}")
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load the trained models
    catboost_model = joblib.load(dataset_info["catboost_model_path"])
    xgboost_model = joblib.load(dataset_info["xgboost_model_path"])
    
    # Create a Voting Classifier
    voting_clf = VotingClassifier(
        estimators=[
            ('catboost', catboost_model),
            ('xgboost', xgboost_model)
        ],
        voting='soft'  # Use 'soft' voting for probability-based aggregation
    )
    
    # Train the Voting Classifier
    print(f"Training the Voting Classifier for {dataset_name}...")
    voting_clf.fit(X_train, y_train)
    
    # Evaluate the Voting Classifier
    y_pred = voting_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Voting Classifier Accuracy for {dataset_name}: {accuracy:.2f}")
    
    # Generate a classification report
    report = classification_report(y_test, y_pred)
    print(f"Classification Report for {dataset_name}:")
    print(report)
    
    # Save the Voting Classifier
    voting_model_path = os.path.join(voting_model_dir, f"{dataset_name}_voting_classifier.pkl")
    joblib.dump(voting_clf, voting_model_path)
    print(f"Voting Classifier saved to: {voting_model_path}")