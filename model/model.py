import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from config import settings
from google.cloud import bigquery
import pickle

# ========== Configurações de Logging ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ========== 1. Função de dados ==========
def carregar_dados():
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
        query_parameters=[
            bigquery.ScalarQueryParameter("symbol", "STRING", symbol)
        ]
    )

    df = client.query(query, job_config=job_config).to_dataframe()
    logging.info(f"✔ Dados carregados para o símbolo: {symbol}")
    return df

# ========== 2. Função de Escalonamento ==========
def transformar_colunas(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    return pd.DataFrame(df_scaled, columns=df.columns)


# ========== 2. Função de Desescalonamento ==========
def desescalonar(y, target_col, df):
    scaler = StandardScaler()
    close_index = df.columns.get_loc(target_col)
    close_mean = scaler.mean_[close_index]
    close_std = scaler.scale_[close_index]
    return y * close_std + close_mean


# ========== 3. Função de sequência ==========
def criar_sequencias(df: pd.DataFrame, n_steps: int, target_col: str):
    X, y = [], []
    for i in range(len(df) - n_steps):
        X_seq = df.drop(columns=target_col).iloc[i:i+n_steps].values
        y_val = df.iloc[i+n_steps][target_col]
        X.append(X_seq)
        y.append(y_val)
    return np.array(X), np.array(y)

# ========== 4. Construção do modelo ==========
def construir_modelo(units=50, dropout_rate=0.2, l2_reg=0.01,
                     activation='tanh', learning_rate=0.001,
                     n_steps=10, n_features=5):
    model = Sequential()
    model.add(LSTM(units=units, activation=activation,
                   input_shape=(n_steps, n_features),
                   kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    return model

# ========== 5. Avaliação ==========
def avaliar_modelo(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    logging.info(f"MSE: {mse:.4f}")
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE: {mae:.4f}")
    logging.info(f"MAPE: {mape:.2f}%")
    logging.info(f"R² Score: {r2:.4f}")

# ========== 6. Visualização ==========
def plotar_resultados(y_real, y_previsto):
    plt.figure(figsize=(12, 6))
    plt.plot(y_real, label="Real")
    plt.plot(y_previsto, label="Previsto")
    plt.title("Previsão de Fechamento da AAPL com LSTM")
    plt.legend()
    plt.show()

# ========== 6. Main ==========
def main():
    df = carregar_dados()
    df = df.drop(columns=['Date', 'symbol'])
    target_col = 'AAPL_Close'
    features = df.drop(columns=['AAPL_Close']).columns.tolist()

    # Split treino/teste
    train_df, test_df = train_test_split(df, test_size=0.2, shuffle=False, random_state=23)


    # Escalamento
    train_scaled_df = transformar_colunas(train_df)
    test_scaled_df = transformar_colunas(test_df)

    # Sequências
    n_steps = 10
    X_train_seq, y_train = criar_sequencias(train_scaled_df, n_steps, target_col=0)
    X_test_seq, y_test = criar_sequencias(test_scaled_df, n_steps, target_col=0)

    # GridSearch com EarlyStopping
    param_grid = {
        "model__units": [50, 100],
        "model__dropout_rate": [0.2, 0.3],
        "model__l2_reg": [0.001, 0.01],
        "model__activation": ["tanh", "relu"],
        "model__learning_rate": [0.001, 0.0005,0.0001],
        "model__n_steps": [5,10],  # Grid pode testar mais de um valor
        "model__n_features": [len(features)],
        "batch_size": [16],
        "epochs": [30],
        "verbose": [0]
    }


    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

    regressor = KerasRegressor(model=construir_modelo, callbacks=[early_stopping])

    grid = GridSearchCV(estimator=regressor,
                        param_grid=param_grid,
                        scoring='neg_mean_squared_error',
                        cv=15,
                        verbose=2,
                        n_jobs=-1)

    logging.info("🔧 Iniciando treinamento...")
    grid.fit(X_train_seq, y_train)
    logging.info("✅ Treinamento finalizado.")

    logging.info(f"Melhores hiperparâmetros encontrados: {grid.best_params_}")

    best_model = grid.best_estimator_

    # Predição
    y_pred = best_model.predict(X_test_seq)


    # Reutilizar scaler do treino para desescalonar
    y_pred_real = desescalonar(y_pred, target_col, train_df)
    y_test_real = desescalonar(y_test, target_col, train_df)

    # Avaliação com valores reais
    logging.info("\n📊 Métricas na base de teste:")
    avaliar_modelo(y_test_real, y_pred_real)

    # Plot
    plotar_resultados(y_test_real, y_pred_real)

    # Salvar modelo
    filename = 'best_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(grid, file)
    logging.info("💾 Modelo salvo como best_model.pkl")

if __name__ == "__main__":
    main()
