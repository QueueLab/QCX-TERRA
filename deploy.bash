#!/bin/bash
# deploy.bash - Setup and deployment script for TerraMind ONNX on Azure ML

set -e  # Exit on error

echo "=========================================="
echo "TerraMind Azure ML Deployment Setup"
echo "=========================================="

# ------------------------------------------------------------------
# 1. Create and activate virtual environment
# ------------------------------------------------------------------
echo ""
echo "Step 1: Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate
echo "✓ Virtual environment activated"

# Verify Python
echo ""
echo "Python version:"
python --version
echo "Python path:"
which python

# ------------------------------------------------------------------
# 2. Install dependencies
# ------------------------------------------------------------------
echo ""
echo "Step 2: Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# ------------------------------------------------------------------
# 3. Azure CLI login and configuration
# ------------------------------------------------------------------
echo ""
echo "Step 3: Azure authentication..."
echo ""
echo "Checking Azure CLI authentication status..."

if ! az account show &> /dev/null; then
    echo "Not logged in to Azure. Please log in..."
    az login
else
    echo "✓ Already logged in to Azure"
    echo ""
    echo "Current subscription:"
    az account show --query "{Name:name, SubscriptionId:id}" -o table
fi

# ------------------------------------------------------------------
# 4. Set environment variables
# ------------------------------------------------------------------
echo ""
echo "Step 4: Configuring environment variables..."
echo ""
echo "Please provide the following Azure ML configuration:"
echo ""

# Get subscription ID
if [ -z "$AZURE_SUBSCRIPTION_ID" ]; then
    read -p "Azure Subscription ID: " AZURE_SUBSCRIPTION_ID
    export AZURE_SUBSCRIPTION_ID
fi

# Get resource group
if [ -z "$AZURE_RESOURCE_GROUP" ]; then
    read -p "Azure Resource Group: " AZURE_RESOURCE_GROUP
    export AZURE_RESOURCE_GROUP
fi

# Get workspace name
if [ -z "$AZURE_WORKSPACE_NAME" ]; then
    read -p "Azure ML Workspace Name: " AZURE_WORKSPACE_NAME
    export AZURE_WORKSPACE_NAME
fi

# Get compute name
if [ -z "$AZURE_COMPUTE_NAME" ]; then
    read -p "Azure ML Compute Cluster Name: " AZURE_COMPUTE_NAME
    export AZURE_COMPUTE_NAME
fi

# Set subscription
echo ""
echo "Setting active subscription..."
az account set --subscription "$AZURE_SUBSCRIPTION_ID"
echo "✓ Subscription set"

# Save configuration to .env file
echo ""
echo "Saving configuration to .env file..."
cat > .env << EOF
# Azure ML Configuration
export AZURE_SUBSCRIPTION_ID="$AZURE_SUBSCRIPTION_ID"
export AZURE_RESOURCE_GROUP="$AZURE_RESOURCE_GROUP"
export AZURE_WORKSPACE_NAME="$AZURE_WORKSPACE_NAME"
export AZURE_COMPUTE_NAME="$AZURE_COMPUTE_NAME"

# Optional: Override defaults
# export ENDPOINT_NAME="terramind-onnx-endpoint"
# export DEPLOYMENT_NAME="terramind-onnx-deploy"
# export INSTANCE_TYPE="Standard_DS3_v2"
# export MODEL_NAME="terramind-onnx-model"
# export HF_MODEL_ID="ibm-esa-geospatial/TerraMind-1.0-base"
EOF
echo "✓ Configuration saved to .env"
echo ""
echo "To reuse this configuration in future sessions, run:"
echo "  source .env"

# ------------------------------------------------------------------
# 5. Verify Azure ML workspace access
# ------------------------------------------------------------------
echo ""
echo "Step 5: Verifying Azure ML workspace access..."
if az ml workspace show --name "$AZURE_WORKSPACE_NAME" --resource-group "$AZURE_RESOURCE_GROUP" &> /dev/null; then
    echo "✓ Successfully connected to Azure ML workspace"
else
    echo "✗ Failed to access Azure ML workspace"
    echo "Please verify your credentials and workspace configuration"
    exit 1
fi

# ------------------------------------------------------------------
# 6. Run deployment
# ------------------------------------------------------------------
echo ""
echo "=========================================="
echo "Ready to deploy!"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Subscription: $AZURE_SUBSCRIPTION_ID"
echo "  Resource Group: $AZURE_RESOURCE_GROUP"
echo "  Workspace: $AZURE_WORKSPACE_NAME"
echo "  Compute: $AZURE_COMPUTE_NAME"
echo ""
read -p "Proceed with deployment? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting deployment..."
    python deploy.py
else
    echo ""
    echo "Deployment cancelled."
    echo "To deploy later, run: python deploy.py"
fi
