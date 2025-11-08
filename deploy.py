# deploy.py
import os
import sys
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import (
    Model,
    Environment,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
)
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError


# ------------------------------------------------------------------
# Configuration - reads from environment variables with validation
# ------------------------------------------------------------------
def get_required_env(var_name: str) -> str:
    """Get required environment variable or exit with error."""
    value = os.getenv(var_name)
    if not value or value.startswith("<"):
        print(f"ERROR: Environment variable '{var_name}' is not set or contains placeholder value.")
        print(f"Please set it using: export {var_name}=<your-value>")
        sys.exit(1)
    return value


def get_optional_env(var_name: str, default: str) -> str:
    """Get optional environment variable with default."""
    value = os.getenv(var_name, default)
    if value.startswith("<"):
        return default
    return value


# Required configuration
SUBSCRIPTION_ID = get_required_env("AZURE_SUBSCRIPTION_ID")
RESOURCE_GROUP = get_required_env("AZURE_RESOURCE_GROUP")
WORKSPACE_NAME = get_required_env("AZURE_WORKSPACE_NAME")
COMPUTE_NAME = get_required_env("AZURE_COMPUTE_NAME")

# Optional configuration with defaults
ENDPOINT_NAME = get_optional_env("ENDPOINT_NAME", "terramind-onnx-endpoint")
DEPLOYMENT_NAME = get_optional_env("DEPLOYMENT_NAME", "terramind-onnx-deploy")
INSTANCE_TYPE = get_optional_env("INSTANCE_TYPE", "Standard_DS3_v2")
MODEL_NAME = get_optional_env("MODEL_NAME", "terramind-onnx-model")
ENVIRONMENT_NAME = get_optional_env("ENVIRONMENT_NAME", "hf-onnx-env")

print("=" * 70)
print("Azure ML Deployment Configuration")
print("=" * 70)
print(f"Subscription ID: {SUBSCRIPTION_ID}")
print(f"Resource Group: {RESOURCE_GROUP}")
print(f"Workspace: {WORKSPACE_NAME}")
print(f"Compute: {COMPUTE_NAME}")
print(f"Endpoint: {ENDPOINT_NAME}")
print(f"Deployment: {DEPLOYMENT_NAME}")
print(f"Model: {MODEL_NAME}")
print("=" * 70)

# ------------------------------------------------------------------
# Initialize Azure ML Client
# ------------------------------------------------------------------
try:
    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)
    print("✓ Successfully connected to Azure ML workspace")
except Exception as e:
    print(f"ERROR: Failed to connect to Azure ML workspace: {e}")
    print("\nTroubleshooting:")
    print("1. Run 'az login' to authenticate")
    print("2. Run 'az account set --subscription <subscription-id>' to set subscription")
    print("3. Verify your credentials have access to the workspace")
    sys.exit(1)

# ------------------------------------------------------------------
# 1. Create/Update Environment
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 1: Creating Azure ML Environment")
print("=" * 70)

try:
    # Check if environment already exists
    try:
        existing_env = ml_client.environments.get(name=ENVIRONMENT_NAME, version="1")
        print(f"✓ Environment '{ENVIRONMENT_NAME}' already exists, using existing version")
    except ResourceNotFoundError:
        print(f"Creating new environment '{ENVIRONMENT_NAME}'...")
        
        # Create environment from conda file
        env = Environment(
            name=ENVIRONMENT_NAME,
            description="Environment for HuggingFace to ONNX conversion",
            conda_file="conda.yaml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
        )
        ml_client.environments.create_or_update(env)
        print(f"✓ Environment '{ENVIRONMENT_NAME}' created successfully")
except Exception as e:
    print(f"ERROR: Failed to create environment: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# 2. Run conversion job
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 2: Submitting ONNX Conversion Job")
print("=" * 70)

try:
    conv_job = command(
        code="./",
        command="python convert_to_onnx.py",
        environment=f"{ENVIRONMENT_NAME}@latest",
        compute=COMPUTE_NAME,
        display_name="TerraMind HF → ONNX conversion",
        experiment_name="terramind-onnx-conversion"
    )

    print("Submitting conversion job to Azure ML...")
    submitted = ml_client.jobs.create_or_update(conv_job)
    print(f"✓ Job submitted: {submitted.name}")
    print(f"  Job URL: {submitted.studio_url}")
    
    print("\nWaiting for job to complete (this may take several minutes)...")
    ml_client.jobs.stream(submitted.name)
    
    # Check job status
    job_status = ml_client.jobs.get(submitted.name)
    if job_status.status != "Completed":
        print(f"ERROR: Job failed with status: {job_status.status}")
        sys.exit(1)
    
    print("✓ Conversion job completed successfully")
except Exception as e:
    print(f"ERROR: Failed to run conversion job: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# 3. Get the registered MLflow model
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 3: Retrieving Registered Model")
print("=" * 70)

try:
    registered_model = ml_client.models.get(name=MODEL_NAME, label="latest")
    print(f"✓ Found registered model: {registered_model.name}")
    print(f"  Version: {registered_model.version}")
    print(f"  Model ID: {registered_model.id}")
except Exception as e:
    print(f"ERROR: Failed to retrieve model '{MODEL_NAME}': {e}")
    print("\nThe model should have been registered by convert_to_onnx.py")
    print("Check the conversion job logs for errors.")
    sys.exit(1)

# ------------------------------------------------------------------
# 4. Create or update endpoint
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 4: Creating/Updating Endpoint")
print("=" * 70)

try:
    # Check if endpoint exists
    try:
        existing_endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)
        print(f"✓ Endpoint '{ENDPOINT_NAME}' already exists")
    except ResourceNotFoundError:
        print(f"Creating new endpoint '{ENDPOINT_NAME}'...")
        endpoint = ManagedOnlineEndpoint(
            name=ENDPOINT_NAME,
            description="ONNX inference API for TerraMind HuggingFace model",
            auth_mode="key"
        )
        ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        print(f"✓ Endpoint '{ENDPOINT_NAME}' created successfully")
except Exception as e:
    print(f"ERROR: Failed to create endpoint: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# 5. Deploy model to endpoint
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Step 5: Deploying Model to Endpoint")
print("=" * 70)

try:
    deployment = ManagedOnlineDeployment(
        name=DEPLOYMENT_NAME,
        endpoint_name=ENDPOINT_NAME,
        model=registered_model.id,
        instance_type=INSTANCE_TYPE,
        instance_count=1
    )

    print(f"Deploying model to endpoint (this may take 10-15 minutes)...")
    ml_client.online_deployments.begin_create_or_update(deployment).wait()
    
    # Set traffic to 100% for this deployment
    endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)
    endpoint.traffic = {DEPLOYMENT_NAME: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    
    print(f"✓ Deployment '{DEPLOYMENT_NAME}' completed successfully")
except Exception as e:
    print(f"ERROR: Failed to deploy model: {e}")
    sys.exit(1)

# ------------------------------------------------------------------
# 6. Get endpoint details
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("Deployment Complete!")
print("=" * 70)

try:
    endpoint = ml_client.online_endpoints.get(name=ENDPOINT_NAME)
    keys = ml_client.online_endpoints.get_keys(name=ENDPOINT_NAME)
    
    print(f"\nEndpoint Name: {ENDPOINT_NAME}")
    print(f"Scoring URI: {endpoint.scoring_uri}")
    print(f"Primary Key: {keys.primary_key}")
    print(f"\nTo test the endpoint, update Test.py with:")
    print(f'  scoring_uri = "{endpoint.scoring_uri}"')
    print(f'  key = "{keys.primary_key}"')
    print("\nThen run: python Test.py")
    
    # Save endpoint details to file
    with open("endpoint_details.txt", "w") as f:
        f.write(f"Endpoint Name: {ENDPOINT_NAME}\n")
        f.write(f"Scoring URI: {endpoint.scoring_uri}\n")
        f.write(f"Primary Key: {keys.primary_key}\n")
    print("\n✓ Endpoint details saved to endpoint_details.txt")
    
except Exception as e:
    print(f"WARNING: Could not retrieve endpoint details: {e}")

print("\n" + "=" * 70)
print("Deployment script completed successfully!")
print("=" * 70)
