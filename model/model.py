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

# ========== 1. Função de dados ==========
def carregar_dados():
    """Carrega os dados do BigQuery para o DataFrame."""
    project_id = settings.BQ_PROJECT_ID
    dataset_id = settings.BQ_DATASET_ID
    table_id = settings.BQ_TABLE_ID
    symbol = settings.SYMBOLS[0]
    credentials = settings.GOOGLE_CREDENTIALS

    client = bigquery.Client.from_service_account_json(credentials)

    query = f"""
    SELECT * 
    FROM `{project_id}.{dataset_id}.{table_id}`
    WHERE symbol = @symbol
    ORDER BY Date
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("symbol", "STRING", symbol)]
    )

    df = client.query(query, job_config=job_config).to_dataframe()
    logging.info("✔ Dados carregados para o símbolo: %s", symbol)
    return df


# ========== 2. Função de Escalonamento ==========
def transformar_colunas(df, scaler):
    """Escalona todas as colunas numéricas do DataFrame usando StandardScaler."""
    df_scaled = scaler.fit_transform(df)
    return pd.DataFrame(df_scaled, columns=df.columns)


# ========== 2. Função de Desescalonamento ==========
def desescalonar(y, target_col, df, scaler):
    """Desescalona os valores previstos usando o StandardScaler ajustado nos dados originais."""
    close_index = df.columns.get_loc(target_col)
    close_mean = scaler.mean_[close_index]
    close_std = scaler.scale_[close_index]
    return y * close_std + close_mean


# ========== 3. Função de sequência ==========
def criar_sequencias(df: pd.DataFrame, n_steps: int, target_col: str):
    """Cria sequências de dados para o modelo LSTM."""
    x, y = [], []
    for i in range(len(df) - n_steps):
        x_seq = df.drop(columns=target_col).iloc[i : i + n_steps].values
        y_val = df.iloc[i + n_steps][target_col]
        x.append(x_seq)
        y.append(y_val)
    return np.array(x), np.array(y)


def preparar_dados(df, target_col, n_steps, scaler):
    """Prepara os dados para o modelo LSTM, incluindo escalonamento e criação de sequências."""
    features = df.drop(columns=[target_col]).columns.tolist()
    # Split e escalonamento
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False, random_state=23)
    train_scaled = transformar_colunas(train_df, scaler=scaler)
    test_scaled = transformar_colunas(test_df, scaler=scaler)

    # Sequências
    x_train, y_train = criar_sequencias(train_scaled, n_steps, target_col=target_col)
    x_test, y_test = criar_sequencias(test_scaled, n_steps, target_col=target_col)

    return {
        "features": features,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "train_df": train_df
    }

# ========== 4. Construção do modelo ==========
def construir_modelo(
model_params: dict = {
            "units": 50,
            "dropout_rate": 0.2,
            "l2_reg": 0.001,
            "activation": "tanh",
            "learning_rate": 0.001,
        },
    n_steps=10,
    n_features=5,
):
    """Constrói o modelo LSTM com os hiperparâmetros fornecidos em model_params."""
    model = Sequential()
    model.add(
        LSTM(
            units=model_params["units"],
            activation=model_params["activation"],
            input_shape=(n_steps, n_features),
            kernel_regularizer=l2(model_params["l2_reg"]),
        )
    )
    model.add(Dropout(model_params["dropout_rate"]))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=model_params["learning_rate"]), loss="mse")
    return model


def rodar_gridsearch(x_train, y_train, features, early_stopping):
    """Executa GridSearchCV para encontrar os melhores hiperparâmetros do modelo."""
    param_grid = {
        "model__model_params__units": [50, 100],
        "model__model_params__dropout_rate": [0.2, 0.3],
        "model__model_params__l2_reg": [0.001, 0.01],
        "model__model_params__activation": ["tanh", "relu"],
        "model__model_params__learning_rate": [0.001, 0.0005, 0.0001],
        "model__n_steps": [5, 10],
        "model__n_features": [len(features)],
        "batch_size": [16],
        "epochs": [30],
        "verbose": [0],
    }

    regressor = KerasRegressor(model=construir_modelo, callbacks=[early_stopping])

    grid = GridSearchCV(
        estimator=regressor,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=15,
        verbose=2,
        n_jobs=-1,
    )

    grid.fit(x_train, y_train)
    return grid

# ========== 5. Avaliação ==========
def avaliar_modelo(y_true, y_pred):
    """Avalia o modelo usando várias métricas."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    logging.debug("MSE: %s", mse)
    logging.info("RMSE: %s", rmse)
    logging.info("MAE: %s", mae)
    logging.info("MAPE: %s",mape)
    logging.info("R² Score: %s",r2)


# ========== 6. Visualização ==========
def plotar_resultados(y_real, y_previsto):
    """Plota os resultados reais vs previstos."""
    plt.figure(figsize=(12, 6))
    plt.plot(y_real, label="Real")
    plt.plot(y_previsto, label="Previsto")
    plt.title("Previsão de Fechamento da AAPL com LSTM")
    plt.legend()
    plt.show()


# ========== 6. Main ==========
def main():

    scaler = StandardScaler()
    """Função principal para executar o pipeline de treinamento e avaliação do modelo."""
    logging.info("🔄 Carregando dados...")
    df = carregar_dados()
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

    #with open("best_model.pkl", "wb") as file:
    #    pickle.dump(grid, file)
    #logging.info("💾 Modelo salvo como best_model.pkl")


if __name__ == "__main__":
    main()
