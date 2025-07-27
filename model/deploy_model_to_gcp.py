from google.cloud import storage
import os

# Configurações
PROJECT_ID = "fiap-4"  # seu projeto GCP
BUCKET_NAME = "model_bucket_fiap-4"
MODEL_FILENAME = "best_model.pkl"

def upload_model_to_bucket(project_id, bucket_name, source_file):
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    blob = bucket.blob(source_file)  # o nome do arquivo dentro do bucket será igual ao nome local
    blob.upload_from_filename(source_file)
    
    print(f"✅ Uploaded {source_file} to gs://{bucket_name}/{source_file}")

if __name__ == "__main__":
    # caminho absoluto do arquivo local
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_file = os.path.join(current_dir, MODEL_FILENAME)
    
    if not os.path.exists(local_file):
        raise FileNotFoundError(f"Arquivo {MODEL_FILENAME} não encontrado no diretório {current_dir}")
    
    upload_model_to_bucket(PROJECT_ID, BUCKET_NAME, local_file)
