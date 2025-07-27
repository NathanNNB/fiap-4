import os
from dotenv import load_dotenv

load_dotenv()

BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID")
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID")
BQ_TABLE_ID = os.getenv("BQ_TABLE_ID")
GOOGLE_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Collecting data settings
# You can modify these settings as needed
START_DATE = "2018-01-01"
END_DATE = "2024-07-20"
SYMBOLS = ["AAPL"]  # You can add more symbols as needed
# Example: SYMBOLS = ["AAPL", "GOOGL", "MSFT"]