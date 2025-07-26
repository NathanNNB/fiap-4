import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

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