import get_bigquery_data
import pre_process_data

X, y, scaler = pre_process_data.main(get_bigquery_data.main(), window_size=60)