# deploy.py
import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Model,
    Environment,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    CodeConfiguration,
)
from azure.identity import DefaultAzureCredential

# ------------------------------------------------------------------
# USER SETTINGS – fill these
# ------------------------------------------------------------------
SUBSCRIPTION_ID = "<your-subscription-id>"
RESOURCE_GROUP   = "<your-resource-group>"
WORKSPACE_NAME   = "<your-aml-workspace>"
COMPUTE_NAME     = "<your-compute-cluster>"   # for conversion job
ENDPOINT_NAME    = "hf-onnx-endpoint"
DEPLOYMENT_NAME  = "hf-onnx-deploy"
INSTANCE_TYPE    = "Standard_DS3_v2"
# ------------------------------------------------------------------

credential = DefaultAzureCredential()
ml_client = MLClient(credential, SUBSCRIPTION_ID, RESOURCE_GROUP, WORKSPACE_NAME)

# --------------------------------------------------------------
# 1. Run conversion job (only needed once)
# --------------------------------------------------------------
from azure.ai.ml import command

conv_job = command(
    code="./",
    command="python convert_to_onnx.py",
    environment="hf-onnx-env@latest",
    compute=COMPUTE_NAME,
    display_name="HF → ONNX conversion",
    experiment_name="onnx-conversion"
)

print("Submitting conversion job...")
submitted = ml_client.jobs.create_or_update(conv_job)
ml_client.jobs.stream(submitted.name)
print("Conversion job finished. ONNX files are in the job output.")

# --------------------------------------------------------------
# 2. Register the ONNX model (point to job output)
# --------------------------------------------------------------
# The conversion job outputs to `azureml://jobs/<job_name>/outputs/artifacts/paths/onnx_model`
# Grab the latest job run name from the UI or via:
#   ml_client.jobs.list(parent_job_name="onnx-conversion")[-1].name
JOB_RUN_NAME = "<latest-conversion-job-run-name>"   # <-- replace

model = Model(
    path=f"azureml://jobs/{JOB_RUN_NAME}/outputs/artifacts/paths/onnx_model",
    name="hf-onnx-model",
    description="ONNX export of HF model",
    type="custom_model"
)
registered_model = ml_client.models.create_or_update(model)
print(f"Model registered: {registered_model.id}")

# --------------------------------------------------------------
# 3. Define environment (conda or Docker)
# --------------------------------------------------------------
env = Environment(
    name="hf-onnx-env",
    conda_file="conda.yaml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04"
)

# --------------------------------------------------------------
# 4. Create endpoint
# --------------------------------------------------------------
endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    description="ONNX inference API for HF model",
    auth_mode="key"
)
ml_client.online_endpoints.begin_create_or_update(endpoint).result()

# --------------------------------------------------------------
# 5. Deploy
# --------------------------------------------------------------
deployment = ManagedOnlineDeployment(
    name=DEPLOYMENT_NAME,
    endpoint_name=ENDPOINT_NAME,
    model=registered_model,
    environment=env,
    code_configuration=CodeConfiguration(
        code="./",                 # local folder with score.py
        scoring_script="score.py"
    ),
    instance_type=INSTANCE_TYPE,
    instance_count=1
)

print("Deploying...")
ml_client.online_deployments.begin_create_or_update(deployment).result()
ml_client.online_endpoints.begin_start(endpoint_name=ENDPOINT_NAME).result()
print(f"Endpoint ready: {ENDPOINT_NAME}")