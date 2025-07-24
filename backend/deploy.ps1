# Set variables
$REGION = "us-central1"
# Verify if the project is set correctly at the gcloud config
$PROJECT_ID = (gcloud config get-value project)
$IMAGE_NAME = "flask-app-fiap-4"
$REPO_NAME = "flask-repo-fiap-4"

Write-Host "Selected project: $PROJECT_ID"
Write-Host "Region: $REGION"
Write-Host "Image name: $IMAGE_NAME"

# Create image repository (ignore error if it already exists)
Write-Host "Creating repository (if needed)..."
gcloud artifacts repositories create $REPO_NAME `
  --repository-format=docker `
  --location=$REGION `
  --description="Docker repository for Flask app"

# Build and push Docker image
Write-Host "Building the image and pushing to Artifact Registry..."
# The Dockerfile must be located in the current working directory.
# The 'gcloud builds submit' command will automatically look for a file named 'Dockerfile' 
# in this directory when building the container image.
# Make sure you run this script from the directory where the Dockerfile is located,
# or adjust the path accordingly.
gcloud builds submit --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME" .

# Deploy to Cloud Run
Write-Host "Deploying to Cloud Run..."
gcloud run deploy flask-service `
  --image "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME" `
  --platform managed `
  --region $REGION `
  --set-env-vars FLASK_ENV=production `
  --allow-unauthenticated
