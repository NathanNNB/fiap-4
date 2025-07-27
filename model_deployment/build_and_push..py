import subprocess

def run_cmd(cmd):
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def create_repo():
    try:
        subprocess.run(
            "gcloud artifacts repositories create vertex-models --repository-format=docker --location=us-central1 --project=fiap-4",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print("Repository created successfully.")
    except subprocess.CalledProcessError as e:
        if "ALREADY_EXISTS" in e.stderr:
            print("Repository already exists, skipping creation.")
        else:
            print("Error creating repository:")
            print(e.stderr)
            raise

def main():
    create_repo()
    run_cmd("gcloud auth configure-docker us-central1-docker.pkg.dev")
    run_cmd("gcloud builds submit --region=us-central1 --config=cloudbuild.yaml .")

if __name__ == "__main__":
    main()
