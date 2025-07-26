"""Este script define as funções utilizadas para construir um modelo LSTM com hiperparâmetros configuráveis."""
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor

def criar_modelo(
    model_params: dict = {
        "units": 50,
        "dropout_rate": 0.2,
        "l2_reg": 0.001,
        "activation": "tanh",
        "learning_rate": 0.001
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
    model.compile(
        optimizer=Adam(learning_rate=model_params["learning_rate"]), loss="mse"
    )
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

    regressor = KerasRegressor(model=criar_modelo, callbacks=[early_stopping])

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