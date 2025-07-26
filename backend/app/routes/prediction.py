import os
from flask import Blueprint, jsonify, request
from flask_cors import CORS
import numpy as np
import joblib

prediction = Blueprint("prediction", __name__)
CORS(prediction)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "best_model.pkl")
# model = joblib.load(MODEL_PATH)

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
        prediction_value = [123.45]


        return jsonify({"predicted_price": float(prediction_value[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# http://192.168.0.32:8080/prediction/


# curl -X POST http://192.168.0.32:8080/prediction/ -H "Content-Type: application/json" -d '{"close_prices": [130.5, 131.2, 132.0, 133.1, 134.5, 135.0, 136.2, 137.8, 138.9, 139.7, 140.5, 141.2, 142.0, 143.1, 144.3, 145.7, 146.8, 147.9, 148.5, 149.3, 150.1, 151.0, 152.4, 153.2, 154.0, 155.1, 156.3, 157.0, 158.5, 159.2, 160.0, 161.1, 162.5, 163.2, 164.0, 165.4, 166.0, 167.1, 168.3, 169.0, 170.2, 171.0, 172.5, 173.4, 174.2, 175.5, 176.0, 177.1, 178.3, 179.2, 180.0, 181.4, 182.0, 183.1, 184.3, 185.0, 186.5, 187.2, 188.8, 189.9]}'
