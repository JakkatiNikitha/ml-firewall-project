from sklearn.ensemble import VotingClassifier
import joblib
import numpy as np

class ClassificationService:
    def __init__(self):
        # Load the trained models
        self.malware_model = joblib.load('src/models/malware_detection_model.pkl')
        self.password_model = joblib.load('src/models/password_attack_model.pkl')
        self.mitm_model = joblib.load('src/models/mitm_attack_model.pkl')

        # Create a voting classifier
        self.voting_classifier = VotingClassifier(
            estimators=[
                ('malware', self.malware_model),
                ('password', self.password_model),
                ('mitm', self.mitm_model)
            ],
            voting='hard'
        )

    def classify(self, data):
        # Make predictions using the voting classifier
        predictions = self.voting_classifier.predict(data)
        return self.aggregate_results(predictions)

    def aggregate_results(self, predictions):
        # Aggregate results for each type of attack
        results = {
            'Malware': np.sum(predictions == 0),
            'Password Attacks': np.sum(predictions == 1),
            'MITM Attacks': np.sum(predictions == 2)
        }
        return results