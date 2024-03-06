from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('fraud_detection_model_with_smote.h5')

# Load the scaler
with open('scaler_with_smote.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def index():
    return render_template('front.html')

@app.route('/app')
def fraud_detection():
    return render_template('fraud-transaction1.html')

    # Define a function to process the uploaded CSV file
def process_uploaded_file(file):
        try:
            # Read the CSV file
            data = pd.read_csv(file)
            
            # Check if the data is empty
            if data.empty:
                return "Error: Empty file uploaded"

            # Scale the input data
            scaled_input = scaler.transform(data.values)

            # Make predictions
            predictions = model.predict(scaled_input)

            # Return prediction results
            return "Fraudulent" if predictions[0][0] >= 0.5 else "Not Fraudulent"
        except Exception as e:
            return f"Error: {str(e)}"
    

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    # Check if request contains file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']

    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Process the uploaded file and make predictions
    prediction = process_uploaded_file(file)

    return jsonify({'results': prediction})
    
    


@app.route('/app1', methods=['GET', 'POST'])
def test_data():
    if request.method == 'POST':
        # Check if request contains file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']

        # Check if file is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        try:
            # Read the CSV file
            data = pd.read_csv(file)
            
            # Check if the data is empty
            if data.empty:
                return jsonify({'error': 'Empty file uploaded'})

            # Split the dataset into features and target variable
            X = data.drop('Class', axis=1)
            y = data['Class']

            # Standardize the input data
            scaled_input = scaler.transform(X)

            # Make predictions
            predictions = (model.predict(scaled_input) >= 0.5).astype(int)

            # Calculate accuracy
            accuracy = np.mean(predictions.flatten() == y.values) * 100

            response_data = {
                'accuracy': float(accuracy),
                'fraudulent_cases': int(predictions.sum()),
                'genuine_cases': int(len(predictions) - predictions.sum()),
                'fraudulent_ratio': float(predictions.sum() / len(predictions)),
                'precision_score': float(np.mean(predictions == y.values)),
                'test_loss': 0.0,  # Placeholder for now
                'test_accuracy': float(accuracy)
            }

            print(response_data)  # Print the response data in the console

            return jsonify(response_data)  # Convert to float

        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('fraud-transaction.html')

if __name__ == '__main__':
    app.run(debug=True)
