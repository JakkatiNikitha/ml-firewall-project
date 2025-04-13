from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import os
import logging
import random

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths for models and datasets
project_root = "C:/Users/Lenovo/OneDrive/Desktop/ml-firewall-project"
models = {
    "mitm_dataset": {
        "model_path": os.path.join(project_root, "backend/src/models/mitm_dataset_voting_classifier.pkl"),
        "label_encoder_path": os.path.join(project_root, "backend/src/models/mitm_label_encoder.pkl"),
    },
    "malware_dataset": {
        "model_path": os.path.join(project_root, "backend/src/models/malware_dataset_voting_classifier.pkl"),
        "label_encoder_path": os.path.join(project_root, "backend/src/models/malware_label_encoder.pkl"),
    },
    "password_dataset": {
        "model_path": os.path.join(project_root, "backend/src/models/password_dataset_voting_classifier.pkl"),
        "label_encoder_path": os.path.join(project_root, "backend/src/models/password_label_encoder.pkl"),
    }
}

@app.route('/')
def index():
    """Redirect to the upload page."""
    return render_template('upload.html', title="Upload Dataset")

@app.route('/upload')
def upload_page():
    """Render the upload dataset page."""
    return render_template('upload.html', title="Upload Dataset")

@app.route('/random_sample_page')
def random_sample_page():
    """Render the analyze random sample page."""
    return render_template('random_sample_page.html', title="Analyze Random Sample")

@app.route('/detect', methods=['POST'])
def detect():
    """Handle file upload and detect attacks for multiple models."""
    logging.info("File upload request received.")

    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read the uploaded CSV file
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Failed to process the uploaded file. Please ensure it is a valid CSV file. Error: {str(e)}'}), 400

    # Define column ranges for each attack type
    column_ranges = {
        "malware_dataset": list(data.columns[:10]),  # First 10 columns
        "mitm_dataset": list(data.columns[10:20]),  # Columns 11-20
        "password_dataset": list(data.columns[20:29])  # Columns 21-29
    }

    # Initialize a dictionary to store the prediction counts
    total_prediction_counts = {}

    # Process each dataset subset
    for dataset, columns in column_ranges.items():
        if dataset not in models:
            continue

        # Load the appropriate model and label encoder
        model_info = models[dataset]
        voting_clf = joblib.load(model_info["model_path"])
        try:
            label_encoder = joblib.load(model_info["label_encoder_path"])
            target_classes = label_encoder.classes_  # Get class names from the LabelEncoder
            logging.info(f"Class names for {dataset}: {target_classes}")
        except FileNotFoundError:
            return jsonify({'error': f'LabelEncoder file not found for {dataset}. Please ensure it exists.'}), 500

        # Ensure the subset has the required columns
        required_columns = voting_clf.feature_names_in_
        subset_data = data[columns]

        # Add missing columns with default values
        for col in required_columns:
            if col not in subset_data.columns:
                subset_data[col] = 0

        # Align columns to match the model's expected input
        X = subset_data[required_columns]

        # Predict using the Voting Classifier
        predictions = voting_clf.predict(X)

        # Count the occurrences of each class
        prediction_counts = pd.Series(predictions).value_counts().to_dict()

        # Map numeric predictions to class names
        prediction_counts_named = {target_classes[int(k)]: v for k, v in prediction_counts.items()}

        # Add the counts to the total
        total_prediction_counts[dataset] = prediction_counts_named

    logging.info(f"Total prediction counts: {total_prediction_counts}")
    return render_template('results.html', prediction_counts=total_prediction_counts)

@app.route('/random_sample', methods=['POST'])
def random_sample():
    """Take a random sample from the uploaded dataset and check for attacks."""
    logging.info("Random sample request received.")

    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Read the uploaded CSV file
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Failed to process the uploaded file. Please ensure it is a valid CSV file. Error: {str(e)}'}), 400

    # Take a random sample from the dataset
    try:
        random_sample = data.sample(n=1, random_state=random.randint(0, 1000))
    except ValueError as e:
        return jsonify({'error': f'Failed to take a random sample. Ensure the dataset is not empty. Error: {str(e)}'}), 400

    # Initialize a dictionary to store the detection results
    detection_results = {}

    # Process the random sample for each dataset
    for dataset, model_info in models.items():
        try:
            # Load the appropriate model and label encoder
            voting_clf = joblib.load(model_info["model_path"])
            label_encoder = joblib.load(model_info["label_encoder_path"])
            target_classes = label_encoder.classes_
        except FileNotFoundError as e:
            logging.error(f"Model or LabelEncoder file not found for {dataset}: {str(e)}")
            return jsonify({'error': f'Model or LabelEncoder file not found for {dataset}. Please ensure it exists.'}), 500
        except Exception as e:
            logging.error(f"Error loading model or LabelEncoder for {dataset}: {str(e)}")
            return jsonify({'error': f'Error loading model or LabelEncoder for {dataset}. Error: {str(e)}'}), 500

        # Ensure the sample has the required columns
        required_columns = voting_clf.feature_names_in_
        sample_data = random_sample.copy()

        # Add missing columns with default values
        for col in required_columns:
            if col not in sample_data.columns:
                sample_data[col] = 0

        # Align columns to match the model's expected input
        try:
            X = sample_data[required_columns]
        except KeyError as e:
            logging.error(f"Missing required columns for {dataset}: {str(e)}")
            return jsonify({'error': f'Missing required columns for {dataset}. Error: {str(e)}'}), 400

        # Predict using the Voting Classifier
        try:
            prediction = voting_clf.predict(X)[0]
            prediction_class = target_classes[int(prediction)]
        except Exception as e:
            logging.error(f"Error during prediction for {dataset}: {str(e)}")
            return jsonify({'error': f'Error during prediction for {dataset}. Error: {str(e)}'}), 500

        # Store the result
        detection_results[dataset] = prediction_class

    # Determine if any attack is detected and collect attack types
    attack_detected = False
    detected_attacks = []
    for dataset, result in detection_results.items():
        if result.lower() != 'normal':  # Assuming 'normal' indicates no attack
            attack_detected = True
            detected_attacks.append(f"{result} ({dataset.replace('_', ' ').title()})")

    # Render the result in the UI
    return render_template(
        'random_sample.html',
        detection_results=detection_results,
        attack_detected=attack_detected,
        detected_attacks=detected_attacks
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)