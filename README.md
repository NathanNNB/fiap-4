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
