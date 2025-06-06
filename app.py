from flask import Flask, render_template, request
import pickle
import numpy as np
import logging
import os

# Initialize Flask app
app = Flask(__name__)

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)

# Load the kidney model
MODEL_PATH = 'C:/Users/HP/Downloads/kidney disease/Python Notebooks/kidney.pkl'
kidney_model = None

try:
    with open(MODEL_PATH, 'rb') as file:
        kidney_model = pickle.load(file)
    logging.info("Kidney model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading kidney model: {e}")

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/predict", methods=['POST'])
def predictPage():
    try:
        to_predict_dict = request.form.to_dict()
        logging.info("Received form data: %s", to_predict_dict)

        # Convert input values to float
        to_predict_list = list(map(float, to_predict_dict.values()))
        logging.info("Converted input data: %s", to_predict_list)

        # Validate feature length (Kidney model expects 18 features)
        if len(to_predict_list) != 18:
            raise ValueError(f"Expected 18 features, got {len(to_predict_list)}")

        # Make prediction
        values = np.asarray(to_predict_list).reshape(1, -1)
        prediction = kidney_model.predict(values)[0]
        
        return render_template('predict.html', pred=prediction)

    except Exception as e:
        logging.error("Prediction error: %s", str(e))
        message = f"Please enter valid data. Error: {str(e)}"
        return render_template("kidney.html", message=message)

if __name__ == '__main__':
    app.run(debug=True)
