from google.cloud import aiplatform

PROJECT_ID = "fiap-4"
REGION = "us-central1"

MODEL_DISPLAY_NAME = "modelo-lstm-pkl"
MODEL_ARTIFACT_URI = "gs://model_bucket_fiap-4/models/"
SERVING_CONTAINER_IMAGE_URI = "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"

ENDPOINT_DISPLAY_NAME = "endpoint-lstm-pkl"

def upload_model():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    model = aiplatform.Model.upload(
        display_name=MODEL_DISPLAY_NAME,
        artifact_uri=MODEL_ARTIFACT_URI,
        serving_container_image_uri=SERVING_CONTAINER_IMAGE_URI,
    )
    print(f"Modelo carregado com ID: {model.resource_name}")
    return model

def create_endpoint():
    endpoint = aiplatform.Endpoint.create(
        display_name=ENDPOINT_DISPLAY_NAME,
        project=PROJECT_ID,
        location=REGION,
    )
    print(f"Endpoint criado com ID: {endpoint.resource_name}")
    return endpoint

def deploy_model_to_endpoint(model, endpoint):
    endpoint.deploy(
        model=model,
        traffic_split={"0": 100},
        deployed_model_display_name=MODEL_DISPLAY_NAME,
    )
    print("Modelo deployado no endpoint.")

if __name__ == "__main__":
    model = upload_model()
    endpoint = create_endpoint()
    deploy_model_to_endpoint(model, endpoint)
