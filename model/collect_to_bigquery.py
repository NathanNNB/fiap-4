import yfinance as yf
import pandas as pd

from google.cloud import bigquery
from config import settings

def colect_data(symbol: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, group_by='ticker')
    
    # Flatten column MultiIndex, if exists
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    
    df.reset_index(inplace=True)  # Reset index to have 'Date' as a column
    df["symbol"] = symbol
    
    return df

def save_bigquery(df: pd.DataFrame, table_id: str):
    client = bigquery.Client.from_service_account_json(settings.GOOGLE_CREDENTIALS)
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE", # Overwrite the table to not create duplicated data
        autodetect=True,
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()
    print(f"✔ Data saved at table: {table_id}")

def main():
    table_completed = f"{settings.BQ_PROJECT_ID}.{settings.BQ_DATASET_ID}.{settings.BQ_TABLE_ID}"
    
    for symbol in settings.SYMBOLS:
        print(f"📈 Collecting data for {symbol}...")
        df = colect_data(symbol, settings.START_DATE, settings.END_DATE)
        print(f"💾 Saving data at BigQuery for {symbol}...")
        save_bigquery(df, table_completed)

if __name__ == "__main__":
    main()
