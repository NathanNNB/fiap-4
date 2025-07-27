"""Este script contém funções para transformar dados, escalonar colunas numéricas, desescalonar valores previstos e criar sequências de dados para modelos LSTM."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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