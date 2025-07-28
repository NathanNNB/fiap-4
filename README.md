# 📈 Stock Closing Price Prediction with Machine Learning

This project aims to predict the closing price of Apple (AAPL) stock based on historical data. We use the `yfinance` library to collect market data, store it in Google BigQuery, and use it to train a predictive model. Additionally, we developed a Flask API that allows predictions based on new historical data, returning both the predicted value and the actual closing price.

The API is currently deployed and accessible at:  
[https://flask-service-8668868710.us-central1.run.app/](https://flask-service-8668868710.us-central1.run.app/)

---

## 🚀 Features

- 📊 Automated collection of Apple stock historical data using `yfinance`
- ☁️ Storage and querying of data in **Google BigQuery**
- 🧠 Training a Machine Learning model to predict closing prices
- 🔌 REST API built with **Flask**, deployed on **Google Cloud Run**, to make predictions from supplied data
- 📈 Returns predicted value alongside actual value for comparison

---
## 🧠 How the Model Was Created

### 1. Data Collection
We used the [`yfinance`](https://pypi.org/project/yfinance/) library to collect historical data for AAPL stock. The data was stored in Google BigQuery, enabling scalable and efficient data querying for analysis and modeling.

### 2. Preprocessing
- Removed irrelevant columns and handled missing values.
- Normalized data to improve model performance through StandardScaler.
- Created time windows (sequences of previous days) as input for the LSTM model.

### 3. Model Construction
- Used an **LSTM (Long Short-Term Memory)** model, well-suited for time series data.
- The model takes a sequence of previous days as input and predicts the next closing price.
- The `scikeras` library was used to integrate Keras models with scikit-learn tools using GridSearch and Cross-Validation to tune hyperparameters and increase performance
### 4. Training
- Split data into training and testing sets.
- Tuned hyperparameters such as number of neurons, dropout rate, and window size.
- Trained locally and exported the model for production use.

### 5. Evaluation
- Evaluated model performance using **MSE (Mean Squared Error)** and **MAE (Mean Absolute Error)**.
- Compared predicted values with actual closing prices.
- Latest metrics achieved: 
- MSE: 16.1684
- RMSE: 4.0210
- MAE: 3.0352
- MAPE: 1.62%
- R² Score: 0.9190

Values using only optimal hyperparameters
<img width="988" height="528" alt="image" src="https://github.com/user-attachments/assets/fc632243-8bcd-43cf-a644-1bdb6e20f109" />


Values with Keras Regressor, GridSearch and Cross-Validation
<img width="1200" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/0b2c2f8e-9288-455f-9d8f-bbdfa9312ccc" />

---


## 🔌 Prediction API
The API is developed with Flask and deployed as a container on **Google Cloud Run** for scalable serving.


- **Base URL:** [https://flask-service-8668868710.us-central1.run.app/](https://flask-service-8668868710.us-central1.run.app/)
- **Route:** `/predict`
- **Method:** `POST`
- **Input Requirements:** JSON payload containing *at least 11 historical records* of stock data. This minimum sequence length is required for the LSTM model to generate a prediction.
- **Sample request:**
<details>
  <summary>Mostrar exemplo de dados JSON</summary>
  ``` json
   [
  {
    "Date": "2025-07-03",
    "AAPL_Open": 212.149994,
    "AAPL_High": 214.649994,
    "AAPL_Low": 211.809998,
    "AAPL_Close": 213.550003,
    "AAPL_Volume": 34955800,
    "symbol": "AAPL"
  },
  {
    "Date": "2025-07-07",
    "AAPL_Open": 212.679993,
    "AAPL_High": 216.229996,
    "AAPL_Low": 208.800003,
    "AAPL_Close": 209.949997,
    "AAPL_Volume": 50229000,
    "symbol": "AAPL"
  },
  {
    "Date": "2025-07-08",
    "AAPL_Open": 210.100006,
    "AAPL_High": 211.429993,
    "AAPL_Low": 208.449997,
    "AAPL_Close": 210.009995,
    "AAPL_Volume": 42848900,
    "symbol": "AAPL"
  },
  {
    "Date": "2025-07-09",
    "AAPL_Open": 209.529999,
    "AAPL_High": 211.330002,
    "AAPL_Low": 207.220001,
    "AAPL_Close": 211.139999,
    "AAPL_Volume": 48749400,
    "symbol": "AAPL"
  },
  {
    "Date": "2025-07-10",
    "AAPL_Open": 210.509995,
    "AAPL_High": 213.479996,
    "AAPL_Low": 210.029999,
    "AAPL_Close": 212.410004,
    "AAPL_Volume": 44443600,
    "symbol": "AAPL"
  },
  {
    "Date": "2025-07-11",
    "AAPL_Open": 210.570007,
    "AAPL_High": 212.130005,
    "AAPL_Low": 209.860001,
    "AAPL_Close": 211.160004,
    "AAPL_Volume": 39765800,
    "symbol": "AAPL"
  },
  {
    "Date": "2025-07-14",
    "AAPL_Open": 209.929993,
    "AAPL_High": 210.910004,
    "AAPL_Low": 207.539993,
    "AAPL_Close": 208.619995,
    "AAPL_Volume": 38840100,
    "symbol": "AAPL"
  },
  {
    "Date": "2025-07-15",
    "AAPL_Open": 209.220001,
    "AAPL_High": 211.889999,
    "AAPL_Low": 208.919998,
    "AAPL_Close": 209.110001,
    "AAPL_Volume": 42296300,
    "symbol": "AAPL"
  },
  {
    "Date": "2025-07-16",
    "AAPL_Open": 210.300003,
    "AAPL_High": 212.399994,
    "AAPL_Low": 208.639999,
    "AAPL_Close": 210.160004,
    "AAPL_Volume": 47490500,
    "symbol": "AAPL"
  },
  {
    "Date": "2025-07-17",
    "AAPL_Open": 210.570007,
    "AAPL_High": 211.800003,
    "AAPL_Low": 209.589996,
    "AAPL_Close": 210.020004,
    "AAPL_Volume": 48068100,
    "symbol": "AAPL"
  },
  {
    "Date": "2025-07-18",
    "AAPL_Open": 210.869995,
    "AAPL_High": 211.789993,
    "AAPL_Low": 209.699997,
    "AAPL_Close": 211.179993,
    "AAPL_Volume": 48974600,
    "symbol": "AAPL"
  }
  ]
  </details>
  ```
- **Sample Response:**
  ```json
  {
    "predicted_value": 189.50,
    "actual_value": 190.02
  }

- Postman Example
<img width="2554" height="1073" alt="Screenshot 2025-07-28 193224" src="https://github.com/user-attachments/assets/ffeb8e3c-a7ea-4646-a34a-94fb37197409" />



---

## 🚀 Deployment to Google Cloud Platform (GCP)
This project includes a PowerShell script (deploy.ps1) to automate the deployment process to Google Cloud Run using Google Artifact Registry and Cloud Build.

### How to use
- Make sure you have the Google Cloud SDK installed and authenticated (gcloud auth login).

- Ensure your current gcloud project is set correctly (gcloud config set project YOUR_PROJECT_ID).

- Run the deploy.ps1 script from the directory containing the Dockerfile.

- The script will build the Docker image, push it to Google Artifact Registry, and deploy it to Cloud Run.

- Once deployed, your Flask API will be available at the Cloud Run URL.

---
### Running local: Model Creation

1. Go to the `model` folder:

   ```bash
   cd model

2. Create and activate a virtual environment:

   ```bash
   source venv/bin/activate (Mac)

   ```bash
   .\venv\Scripts\activate (Windows)

3. Install the requirements:

   ```bash
   pip install -r requirements.txt

4. Create the .env file with the variables:

    ```bash
    BQ_PROJECT_ID # Verify the project ID at the GCP 
    BQ_DATASET_ID # Verify the BigQuery Dataset ID at the GCP 
    BQ_TABLE_ID # If the table does not exists at the dataset, a new table will be created 
    GOOGLE_APPLICATION_CREDENTIALS # Create or get the credentials to be able to perform actions with the GCP


5. Configure the settings.py, at this project the Apple financial records will be used for the model training:

    ```bash
    START_DATE = "2018-01-01"
    END_DATE = "2024-07-20"
    SYMBOLS = ["AAPL"]  # You can add more symbols as needed

---
### Running local: Backend

1. Open the backend folder:
   ```
   bash
   cd backend
   ```
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # (Linux/Mac) or 
   .\venv\Scripts\activate (Windows)
   ```
4. Install the requirements:
   ```
   pip install -r requirements.txt
   ```
6. Run the `main.py` file:
   ```
   python main.py
   ```

--- 

## 👥 Authors

- [Nathan Novais Borges](https://github.com/NathanNNB)  
- [William Bilatto](https://github.com/WilliamBilatto)  
