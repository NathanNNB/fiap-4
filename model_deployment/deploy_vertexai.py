from google.cloud import aiplatform

PROJECT_ID = "fiap-4"
REGION = "us-central1"

MODEL_DISPLAY_NAME = "scikeras-model"
ENDPOINT_DISPLAY_NAME = "scikeras-endpoint"
CONTAINER_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/vertex-models/scikeras-predictor"
MODEL_ENDPOINT = "gs://model_bucket_fiap-4/models/"

def create_model():
    aiplatform.init(project=PROJECT_ID, location=REGION)
    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=MODEL_ENDPOINT,  # Diretório do modelo no bucket
        serving_container_image_uri=CONTAINER_IMAGE_URI,  # Imagem docker para servir o modelo
    )
    print(f"Model created: {model.resource_name}")
    return model

def create_endpoint():
    aiplatform.init(project=PROJECT_ID, location=REGION)
    endpoint = aiplatform.Endpoint.create(display_name=ENDPOINT_DISPLAY_NAME)
    print(f"Endpoint created: {endpoint.resource_name}")
    return endpoint

def deploy_model_to_endpoint(model, endpoint):
    deployed_model = endpoint.deploy(
        model=model,
        deployed_model_display_name="scikeras-deployment",
        machine_type="n1-standard-2",
        traffic_percentage=100,
    )
    print(f"Model deployed: {deployed_model.resource_name}")

def main():
    model = create_model()
    endpoint = create_endpoint()
    deploy_model_to_endpoint(model, endpoint)

if __name__ == "__main__":
    main()
