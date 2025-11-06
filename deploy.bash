# 1. Clone / create folder
mkdir hf-onnx-azure && cd hf-onnx-azure

# 2. Save the 5 files above (copy-paste)

# 3. Install SDK locally (optional, for testing)
pip install azure-ai-ml azure-identity

# 4. Login
az login
az account set --subscription <your-subscription-id>

# 5. Run deployment script
python deploy.py
