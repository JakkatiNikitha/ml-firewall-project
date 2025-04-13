# Machine Learning-Based Firewall Project - Backend

This project implements a machine learning-based firewall system designed to intelligently detect and classify cyberattacks, specifically focusing on three major categories: Malware, Password-based Attacks, and Man-in-the-Middle (MITM) Attacks.

## Project Structure

- **src/**: Contains the main application code.
  - **app.py**: The entry point for the backend application. Initializes the Flask app and sets up middleware and routes.
  - **models/**: Directory containing trained machine learning models for detecting various types of attacks.
    - **malware_detection_model.pkl**: Model for detecting malware.
    - **password_attack_model.pkl**: Model for identifying password-based attacks.
    - **mitm_attack_model.pkl**: Model for detecting Man-in-the-Middle attacks.
  - **routes/**: Contains API endpoint definitions.
    - **api.py**: Handles incoming requests and routes them to the appropriate service.
  - **services/**: Contains the logic for detecting cyberattacks.
    - **detection_service.py**: Logic for detecting various types of attacks using the models.
    - **classification_service.py**: Additional classification functionalities for complex scenarios.
  - **utils/**: Utility functions for data preprocessing.
    - **preprocessing.py**: Functions for cleaning and preparing data for model input.

- **requirements.txt**: Lists the dependencies required for the backend application, including Flask and machine learning libraries.

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd ml-firewall-project/backend
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
   Then install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Start the Flask application by running:
   ```
   python src/app.py
   ```

4. **Access the API**:
   The API will be available at `http://localhost:5000`. You can use tools like Postman or curl to interact with the endpoints defined in `api.py`.

## Usage Guidelines

- The backend is designed to accept datasets for classification. Users can upload datasets through the frontend interface, and the backend will process them to identify potential threats.
- Ensure that the models are trained and available in the `models/` directory before running the application.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.