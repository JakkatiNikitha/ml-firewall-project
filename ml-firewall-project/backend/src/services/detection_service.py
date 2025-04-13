from flask import jsonify
import pandas as pd
import joblib

class DetectionService:
    def __init__(self):
        self.malware_model = joblib.load('backend/src/models/malware_detection_model.pkl')
        self.password_model = joblib.load('backend/src/models/password_attack_model.pkl')
        self.mitm_model = joblib.load('backend/src/models/mitm_attack_model.pkl')

    def detect_attacks(self, data):
        # Preprocess the incoming data
        processed_data = self.preprocess_data(data)

        # Make predictions using the models
        malware_predictions = self.malware_model.predict(processed_data)
        password_predictions = self.password_model.predict(processed_data)
        mitm_predictions = self.mitm_model.predict(processed_data)

        # Aggregate results
        results = {
            'malware': sum(malware_predictions),
            'password_attacks': sum(password_predictions),
            'mitm_attacks': sum(mitm_predictions)
        }

        return jsonify(results)

    def preprocess_data(self, data):
        # Convert incoming data to DataFrame
        df = pd.DataFrame(data)

        # Implement necessary preprocessing steps
        # For example: feature selection, normalization, etc.
        # This is a placeholder for actual preprocessing logic
        return df