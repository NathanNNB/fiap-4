import os
from flask import Blueprint, jsonify, request
from flask_cors import CORS
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(".."))) 
from utils.tranform_data import criar_sequencias, transformar_colunas, desescalonar
from utils.build_model import criar_modelo

prediction = Blueprint("prediction", __name__)
CORS(prediction)

def carregar_modelo():
    """Carrega o modelo treinado."""
    model_path = "models/modelo_LSTM_v2.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)

def refined_date(json_data, scaler):
    df = pd.read_json(json_data)
    df_scaled = transformar_colunas(df, scaler)
    result = criar_sequencias(df_scaled, n_steps=5, target_col="AAPL_Close")
    return result
model = carregar_modelo()

print("Modelo carregado com sucesso.", model)

@prediction.route("/", methods=["POST"])
def get_predicition():

    data = request.get_json()
    
    if not data:
        return jsonify({"error": "JSON not found"}), 400
    
    #close_prices = data.get("close_prices")
    try:
        # You may need to reshape the input depending on your model
        #input_array = np.array(close_prices[-60:]).reshape(1, -1)
        scaler = StandardScaler()
        df =  pd.DataFrame(data)
        df = df.drop(columns=["Date", "symbol"])
        df_scaled = transformar_colunas(df, scaler)
        result, pred = criar_sequencias(df_scaled, n_steps=10, target_col="AAPL_Close")
        # Make the prediction
        prediction_value = model.predict(result)
        y_pred_real = desescalonar(prediction_value, "AAPL_Close", df, scaler)
        y_test_real = desescalonar(pred, "AAPL_Close", df, scaler)

        #prediction_value = [123.45]


        return jsonify(
            {"predicted_price": float(y_pred_real[0])},
            {"valor_real_price": float(y_test_real[0])}
                )
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# http://192.168.0.32:8080/prediction/


# curl -X POST http://192.168.0.32:8080/prediction/ -H "Content-Type: application/json" -d '{"close_prices": [130.5, 131.2, 132.0, 133.1, 134.5, 135.0, 136.2, 137.8, 138.9, 139.7, 140.5, 141.2, 142.0, 143.1, 144.3, 145.7, 146.8, 147.9, 148.5, 149.3, 150.1, 151.0, 152.4, 153.2, 154.0, 155.1, 156.3, 157.0, 158.5, 159.2, 160.0, 161.1, 162.5, 163.2, 164.0, 165.4, 166.0, 167.1, 168.3, 169.0, 170.2, 171.0, 172.5, 173.4, 174.2, 175.5, 176.0, 177.1, 178.3, 179.2, 180.0, 181.4, 182.0, 183.1, 184.3, 185.0, 186.5, 187.2, 188.8, 189.9]}'
