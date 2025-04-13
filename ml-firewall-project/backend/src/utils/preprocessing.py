from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import random
import os
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings for cleaner output

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def load_data(file_path):
    """Load dataset from a specified file path."""
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    """Clean the dataset by handling missing values, duplicates, garbage values, and outliers."""
    # Remove duplicates
    data = data.drop_duplicates()

    # Handle missing values
    data = data.fillna(method='ffill')  # Forward fill for missing values

    # Identify and remove garbage values (e.g., negative values in numeric columns)
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if (data[col] < 0).any():
            print(f"Removing garbage values in column: {col}")
            data = data[data[col] >= 0]

    # Identify and handle outliers using the IQR method
    Q1 = data[numeric_columns].quantile(0.25)
    Q3 = data[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    for col in numeric_columns:
        lower_bound = Q1[col] - 1.5 * IQR[col]
        upper_bound = Q3[col] + 1.5 * IQR[col]
        outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
        if outliers > 0:
            print(f"Removing {outliers} outliers in column: {col}")
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

    return data

def use_selected_features(data, selected_features, target_column, output_file="data/processed/preprocessed_data.csv"):
    """Use the already selected features for further processing."""
    print("Using the selected features for further processing...")

    # Ensure the selected features exist in the dataset
    missing_features = [feature for feature in selected_features if feature not in data.columns]
    if missing_features:
        raise ValueError(f"The following selected features are missing in the dataset: {missing_features}")

    # Create a reduced dataset with only selected features and the target column
    reduced_data = data[selected_features + [target_column]]

    # Save the reduced dataset to a CSV file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)  # Ensure the directory exists
    reduced_data.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to: {output_file}")

    return reduced_data

def preprocess_data(file_path, selected_features, target_column, output_file):
    """Main function to preprocess data using selected features."""
    # Load and clean the data
    data = load_data(file_path)
    data = clean_data(data)

    # Use the selected features
    reduced_data = use_selected_features(data, selected_features, target_column, output_file)

    return reduced_data