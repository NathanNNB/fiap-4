import os
from flask import Blueprint, jsonify, request
from flask_cors import CORS
import numpy as np
import joblib

prediction = Blueprint("prediction", __name__)
CORS(prediction)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
model = joblib.load(MODEL_PATH)

@prediction.route("/", methods=["POST"])
def get_predicition():

    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON not found"}), 400
    
    close_prices = data.get("close_prices")
    if not close_prices or not isinstance(close_prices, list):
        return jsonify({"error": "close_prices needs to be a list with numerical values"}), 400
    
    if len(close_prices) < 5:
        return jsonify({"error": "Please provide at least 5 price values for prediction"}), 400
    
    try:
        # You may need to reshape the input depending on your model
        input_array = np.array(close_prices[-60:]).reshape(1, -1)

        # Make the prediction
        # prediction_value = model.predict(input_array)
        prediction_value = "sss"


        return jsonify({"predicted_price": float(prediction_value[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

