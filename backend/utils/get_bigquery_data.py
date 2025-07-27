from google.cloud import bigquery
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join("..")))
from config import settings

def main():
    project_id = settings.BQ_PROJECT_ID
    dataset_id = settings.BQ_DATASET_ID
    table_id = settings.BQ_TABLE_ID
    symbol = settings.SYMBOLS[0]  # Assuming you want to fetch data for the first symbol
    credentials = settings.GOOGLE_CREDENTIALS
    print(credentials)
    print(sys.path.append(os.path.abspath(os.path.join(".."))))
    print("Current working directory:", os.getcwd())
    print(os.path.exists(credentials))
    
    
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
    print(df.head())
    print(f"✔ Data fetched for symbol: {symbol}")
    return df

if __name__ == "__main__":
    main()
