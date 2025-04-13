from .preprocessing import load_data, preprocess_data
import os

# Define the paths to the raw datasets
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "C:/Users/Lenovo/OneDrive/Desktop/ml-firewall-project"))
datasets = {
    "mitm_dataset": os.path.join(project_root, "data/raw/All_Labelled.csv"),
    "malware_dataset": os.path.join(project_root, "data/raw/Obfuscated-MalMem2022.csv"),
    "password_dataset": os.path.join(project_root, "data/raw/UNSW_NB15.csv"),
}

# Define the target column for each dataset
target_columns = {
    "mitm_dataset": "Label",  # Replace with the actual target column name for this dataset
    "malware_dataset": "Class",  # Replace with the actual target column name for this dataset
    "password_dataset": "label",  # Replace with the actual target column name for this dataset
}

# Define the selected features for each dataset
selected_features = {
    "mitm_dataset": ['src_port', 'bidirectional_bytes', 'src2dst_bytes',
       'bidirectional_mean_ps', 'bidirectional_stddev_ps',
       'bidirectional_max_ps', 'dst2src_min_ps', 'src2dst_mean_piat_ms',
       'application_name', 'requested_server_name'],  # Replace with the actual selected features for this dataset
    "malware_dataset": ['pslist.avg_handlers', 'dlllist.avg_dlls_per_proc',
       'handles.avg_handles_per_proc', 'handles.nevent', 'handles.nsection',
       'handles.nmutant', 'svcscan.nservices', 'svcscan.kernel_drivers',
       'svcscan.process_services', 'svcscan.shared_process_services'],  # Replace with the actual selected features for this dataset
    "password_dataset": ['sbytes', 'dbytes', 'rate', 'sttl', 'sload', 'dload', 'ct_state_ttl',
       'ct_dst_src_ltm', 'ct_srv_dst', 'attack_cat'],  # Replace with the actual selected features for this dataset
}

# Perform preprocessing for each dataset
for name, path in datasets.items():
    print(f"Processing dataset: {name}")
    
    # Get the target column and selected features for the current dataset
    target_column = target_columns[name]
    features = selected_features[name]
    
    # Define the output file path
    output_file = os.path.join(project_root, f"data/processed/{name}_preprocessed.csv")
    
    # Preprocess the dataset using the selected features
    preprocess_data(
        file_path=path,
        selected_features=features,
        target_column=target_column,
        output_file=output_file
    )
    
    print(f"Preprocessed data saved to: {output_file}")