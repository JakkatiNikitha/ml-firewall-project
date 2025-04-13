from sklearn.preprocessing import LabelEncoder
import joblib

# Example target labels for each dataset
malware_labels = ['malicious', 'benign']
mitm_labels = ['arp_spoofing', 'normal']
password_labels = ['attack', 'normal']

# Create and save LabelEncoder for malware dataset
malware_encoder = LabelEncoder()
malware_encoder.fit(malware_labels)
joblib.dump(malware_encoder, "C:/Users/Lenovo/OneDrive/Desktop/ml-firewall-project/backend/src/models/malware_label_encoder.pkl")

# Create and save LabelEncoder for MITM dataset
mitm_encoder = LabelEncoder()
mitm_encoder.fit(mitm_labels)
joblib.dump(mitm_encoder, "C:/Users/Lenovo/OneDrive/Desktop/ml-firewall-project/backend/src/models/mitm_label_encoder.pkl")

# Create and save LabelEncoder for password dataset
password_encoder = LabelEncoder()
password_encoder.fit(password_labels)
joblib.dump(password_encoder, "C:/Users/Lenovo/OneDrive/Desktop/ml-firewall-project/backend/src/models/password_label_encoder.pkl")