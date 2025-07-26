## Project Description

### 🔧 Model Creation

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

###  Backend

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



# curl -X POST http://localhost:8080/prediction/ -H "Content-Type: application/json" -d '{"close_prices": [130.5, 131.2, 132.0, 133.1, 134.5, 135.0, 136.2, 137.8, 138.9, 139.7, 140.5, 141.2, 142.0, 143.1, 144.3, 145.7, 146.8, 147.9, 148.5, 149.3, 150.1, 151.0, 152.4, 153.2, 154.0, 155.1, 156.3, 157.0, 158.5, 159.2, 160.0, 161.1, 162.5, 163.2, 164.0, 165.4, 166.0, 167.1, 168.3, 169.0, 170.2, 171.0, 172.5, 173.4, 174.2, 175.5, 176.0, 177.1, 178.3, 179.2, 180.0, 181.4, 182.0, 183.1, 184.3, 185.0, 186.5, 187.2, 188.8, 189.9]}'