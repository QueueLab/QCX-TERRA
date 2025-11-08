# QCX-Terra: TerraMind ONNX Deployment on Azure ML

This repository provides scripts and configuration for deploying the TerraMind HuggingFace model as an ONNX inference endpoint on Azure Machine Learning.

## Overview

The deployment process consists of the following steps:

1. **Environment Setup**: Create Azure ML environment with required dependencies
2. **Model Conversion**: Convert HuggingFace model to ONNX format using Azure ML compute
3. **Model Registration**: Register the ONNX model with MLflow in Azure ML
4. **Endpoint Creation**: Create a managed online endpoint for inference
5. **Deployment**: Deploy the model to the endpoint with auto-scaling

## Prerequisites

Before running the deployment, ensure you have:

- **Azure Subscription** with access to Azure Machine Learning
- **Azure ML Workspace** already created
- **Compute Cluster** provisioned in your workspace (CPU or GPU)
- **Azure CLI** installed and configured (`az` command available)
- **Python 3.9+** installed locally
- **Appropriate permissions** to create and manage Azure ML resources

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/queuelab/qcx-terra.git
cd qcx-terra
```

### 2. Run the Deployment Script

The easiest way to deploy is using the provided bash script:

```bash
chmod +x deploy.bash
./deploy.bash
```

This script will:
- Create a Python virtual environment
- Install all required dependencies
- Prompt you for Azure configuration
- Authenticate with Azure
- Run the deployment process

### 3. Manual Deployment (Alternative)

If you prefer manual control, follow these steps:

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Login to Azure
az login
az account set --subscription <your-subscription-id>

# Set environment variables
export AZURE_SUBSCRIPTION_ID="<your-subscription-id>"
export AZURE_RESOURCE_GROUP="<your-resource-group>"
export AZURE_WORKSPACE_NAME="<your-workspace-name>"
export AZURE_COMPUTE_NAME="<your-compute-cluster>"

# Run deployment
python deploy.py
```

## Configuration

The deployment scripts use environment variables for configuration. All required variables must be set before running `deploy.py`.

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_SUBSCRIPTION_ID` | Your Azure subscription ID | `12345678-1234-1234-1234-123456789abc` |
| `AZURE_RESOURCE_GROUP` | Resource group containing your workspace | `my-ml-resources` |
| `AZURE_WORKSPACE_NAME` | Name of your Azure ML workspace | `my-ml-workspace` |
| `AZURE_COMPUTE_NAME` | Name of your compute cluster | `cpu-cluster` |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENDPOINT_NAME` | Name for the inference endpoint | `terramind-onnx-endpoint` |
| `DEPLOYMENT_NAME` | Name for the deployment | `terramind-onnx-deploy` |
| `INSTANCE_TYPE` | VM size for deployment | `Standard_DS3_v2` |
| `MODEL_NAME` | Registered model name | `terramind-onnx-model` |
| `ENVIRONMENT_NAME` | Azure ML environment name | `hf-onnx-env` |
| `HF_MODEL_ID` | HuggingFace model identifier | `ibm-esa-geospatial/TerraMind-1.0-base` |
| `ONNX_OPSET` | ONNX opset version | `14` |

### Saving Configuration

You can save your configuration in a `.env` file:

```bash
# .env
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="your-resource-group"
export AZURE_WORKSPACE_NAME="your-workspace-name"
export AZURE_COMPUTE_NAME="your-compute-cluster"
```

Then source it before deployment:

```bash
source .env
python deploy.py
```

## Testing the Endpoint

After successful deployment, the endpoint details will be saved to `endpoint_details.txt`. You can test the endpoint using the provided test script:

```bash
python Test.py
```

The test script will:
- Load endpoint configuration from `endpoint_details.txt` or environment variables
- Create a sample payload with dummy tensor data
- Send a request to the endpoint
- Display the response

### Manual Testing

You can also test the endpoint manually using curl:

```bash
# Get endpoint details
SCORING_URI="<your-endpoint-uri>"
PRIMARY_KEY="<your-primary-key>"

# Send test request
curl -X POST "$SCORING_URI" \
  -H "Authorization: Bearer $PRIMARY_KEY" \
  -H "Content-Type: application/json" \
  -d '{"S2L2A": "<base64-encoded-tensor>"}'
```

## Project Structure

```
qcx-terra/
├── deploy.py              # Main deployment script
├── deploy.bash            # Automated setup and deployment
├── convert_to_onnx.py     # Model conversion script (runs on Azure ML)
├── Test.py                # Endpoint testing script
├── requirements.txt       # Python dependencies
├── conda.yaml             # Azure ML environment definition
├── terramind_config.yaml  # Model configuration
├── Dockerfile.dockerfile  # Docker configuration (optional)
└── README.md              # This file
```

## Troubleshooting

### Issue: "Environment variable not set"

**Solution**: Ensure all required environment variables are set before running `deploy.py`. Use the `deploy.bash` script for guided setup, or manually export the variables.

### Issue: "Failed to connect to Azure ML workspace"

**Possible causes**:
- Not logged in to Azure: Run `az login`
- Wrong subscription: Run `az account set --subscription <subscription-id>`
- Insufficient permissions: Verify you have Contributor or Owner role on the workspace
- Incorrect workspace details: Double-check resource group and workspace names

### Issue: "Compute cluster not found"

**Solution**: Create a compute cluster in your Azure ML workspace before deployment:

```bash
az ml compute create --name cpu-cluster \
  --type AmlCompute \
  --size Standard_DS3_v2 \
  --min-instances 0 \
  --max-instances 4 \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

### Issue: "Model registration failed"

**Possible causes**:
- Conversion job failed: Check the job logs in Azure ML Studio
- MLflow not properly configured: Ensure `azureml-mlflow` is installed
- Network issues: Verify connectivity to Azure ML

**Solution**: Review the conversion job logs in Azure ML Studio to identify the specific error.

### Issue: "Deployment timeout"

**Solution**: Deployment can take 10-15 minutes. If it times out:
- Check Azure ML Studio for deployment status
- Verify the instance type is available in your region
- Check quota limits for your subscription

### Issue: "Endpoint returns 500 error"

**Possible causes**:
- Model inference error: Check deployment logs in Azure ML Studio
- Incorrect input format: Verify payload structure matches model expectations
- Resource constraints: Try a larger instance type

**Solution**: Review deployment logs and test with the provided `Test.py` script first.

## Advanced Configuration

### Using a Different HuggingFace Model

To deploy a different HuggingFace model, set the `HF_MODEL_ID` environment variable:

```bash
export HF_MODEL_ID="your-org/your-model"
python deploy.py
```

Ensure the model is compatible with the ONNX export process.

### Custom Instance Types

To use a different VM size for deployment:

```bash
export INSTANCE_TYPE="Standard_NC6"  # GPU instance
python deploy.py
```

Available instance types depend on your subscription and region. See [Azure ML VM sizes](https://learn.microsoft.com/en-us/azure/machine-learning/reference-managed-online-endpoints-vm-sku-list) for options.

### Multiple Deployments

To create multiple deployments (e.g., for A/B testing):

```bash
# First deployment
export DEPLOYMENT_NAME="deployment-v1"
python deploy.py

# Second deployment
export DEPLOYMENT_NAME="deployment-v2"
python deploy.py
```

Traffic can be split between deployments in Azure ML Studio.

## Monitoring and Maintenance

### View Endpoint Logs

```bash
az ml online-deployment get-logs \
  --name <deployment-name> \
  --endpoint-name <endpoint-name> \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

### Update Deployment

To update an existing deployment with a new model version:

1. Run the conversion job again to register a new model version
2. Update the deployment to use the new model version
3. The deployment will perform a blue-green update with zero downtime

### Delete Resources

To clean up resources:

```bash
# Delete deployment
az ml online-deployment delete \
  --name <deployment-name> \
  --endpoint-name <endpoint-name> \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>

# Delete endpoint
az ml online-endpoint delete \
  --name <endpoint-name> \
  --resource-group <resource-group> \
  --workspace-name <workspace-name>
```

## Contributing

Contributions are welcome! Please submit issues and pull requests to improve the deployment scripts and documentation.

## License

This project is part of the QCX (Quality Computer Experience) system by QueueLab.

## Support

For issues and questions:
- Open an issue on GitHub
- Check the [Azure ML documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)
- Review the [troubleshooting section](#troubleshooting) above

## References

- [Azure Machine Learning Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [ONNX Documentation](https://onnx.ai/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
- [TerraMind Model](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base)
