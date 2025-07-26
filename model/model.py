"""model.py
This module contains functions for loading data, preprocessing, building
 and evaluating an LSTM model for time series forecasting."""

import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scikeras.wrappers import KerasRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from google.cloud import bigquery
from config import settings
import get_bigquery_data 
from evaluate_model import avaliar_modelo, plotar_resultados
from tranform_data import preparar_dados, desescalonar
from build_model import rodar_gridsearch

# ========== Configurações de Logging ==========
# ==============================
# 1. Configuração do logger principal (do seu código)
# ==============================
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # ou INFO, dependendo da sua necessidade

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

# ==============================
# 2. Reduz o nível de logging de bibliotecas externas
# ==============================
# Isso afeta todas as libs que usam o logger root (como sklearn, pandas, etc.)
for lib in [
    'tensorflow', 'scikeras', 'keras', 'sklearn', 'matplotlib', 'urllib3',
    'google', 'google.cloud', 'pandas', 'numpy'
]:
    logging.getLogger(lib).setLevel(logging.ERROR)

# Isso impede a propagação dos logs dos filhos para o root logger (opcional)
logging.getLogger().setLevel(logging.ERROR)


# ========== 6. Main ==========
def main():

    scaler = StandardScaler()
    """Função principal para executar o pipeline de treinamento e avaliação do modelo."""
    logging.info("🔄 Carregando dados...")
    df = get_bigquery_data.main()
    df = df.drop(columns=["Date", "symbol"])
    target_col = "AAPL_Close"
    n_steps = 10

    dados = preparar_dados(df, target_col, n_steps, scaler)

    early_stopping = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)

    logging.info("🔧 Iniciando treinamento...")
    grid = rodar_gridsearch(dados["x_train"], dados["y_train"], dados["features"], early_stopping)
    logging.info("✅ Treinamento finalizado.")
    logging.info("Melhores hiperparâmetros encontrados: %s", grid.best_params_)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(dados["x_test"])

    y_pred_real = desescalonar(y_pred, target_col, dados["train_df"], scaler)
    y_test_real = desescalonar(dados["y_test"], target_col, dados["train_df"], scaler)

    # Avaliação com valores reais
    logging.info("\n📊 Métricas na base de teste:")
    avaliar_modelo(y_test_real, y_pred_real)

    # Plot
    plotar_resultados(y_test_real, y_pred_real)

    with open("best_model.pkl", "wb") as file:
        pickle.dump(best_model, file)
    logging.info("💾 Modelo salvo como best_model.pkl")


if __name__ == "__main__":
    main()
