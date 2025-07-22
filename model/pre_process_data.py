from sklearn.preprocessing import MinMaxScaler
import numpy as np

def main(df, window_size=60):
    # Use only the closing prices
    close_prices = df["Close"].values.reshape(-1, 1)

    # Normalize prices between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close_prices)

    # Create sequence windows (X) and targets (y)
    # X will be a sequence of prices and y will be the next price
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size:i])
        y.append(scaled[i])
    
    # Convert to numpy arrays and reshape for LSTM input
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # (samples, time steps, features)

    return X, y, scaler

if __name__ == "__main__":
    main()