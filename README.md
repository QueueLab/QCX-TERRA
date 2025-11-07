# QCX-Terra Cloud Model Fine-Tuning and Deployment

This repository contains a complete workflow for fine-tuning, deploying, and scoring the TerraMind model, a powerful geospatial foundation model. The process leverages Azure Machine Learning for robust deployment and management.

## Project Overview

The workflow consists of the following key stages:

1.  **Configuration**: Defining model, data, and training parameters in `terramind_config.yaml`.
2.  **Fine-Tuning (Placeholder)**: A script `finetune_job.py` is included, but the implementation is currently empty. This is where you would add your custom fine-tuning logic using PyTorch Lightning.
3.  **Conversion to ONNX**: Converting the fine-tuned model to the ONNX format for optimized inference using `convert_to_onnx.py`.
4.  **Deployment**: Deploying the ONNX model as a managed online endpoint in Azure Machine Learning using `deploy.py`.

## Getting Started

### Prerequisites

*   An Azure subscription
*   An Azure Machine Learning workspace
*   A compute cluster in your Azure ML workspace
*   Python 3.8 or later
*   The Azure CLI and the ML extension

### Environment Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**

    The required Python packages are listed in `requirements.txt` and the conda environment is defined in `conda.yaml`. You can set up the environment using either pip or conda:

    Using pip:

    ```bash
    pip install -r requirements.txt
    ```

    Using conda:

    ```bash
    conda env create -f conda.yaml
    conda activate terramind
    ```

## Configuration

The `terramind_config.yaml` file is the central place to configure the model, data, and training process.

*   **`model`**: Defines the model architecture, pretrained weights, and modalities.
*   **`data`**: Specifies the paths to your training and validation data, batch size, and other data loading parameters.
*   **`trainer`**: Configures the PyTorch Lightning trainer, including the accelerator, number of devices, and number of epochs.
*   **`callbacks`**: Defines callbacks, such as `ModelCheckpoint`, to save the best model during training.
*   **`logger`**: Configures the logger, such as `TensorBoardLogger`, to log training progress.

## Fine-Tuning

The `finetune_job.py` script is intended for fine-tuning the TerraMind model. The script is currently a placeholder and needs to be implemented. A typical implementation would involve:

1.  Loading the configuration from `terramind_config.yaml`.
2.  Instantiating the `TerraMindDataModule` and the model.
3.  Creating a `Trainer` instance with the specified callbacks and logger.
4.  Calling `trainer.fit()` to start the fine-tuning process.

## Conversion to ONNX

The `convert_to_onnx.py` script converts a trained model from a Hugging Face Hub repository to the ONNX format.

### User Settings

Before running the script, you need to update the following variables in `convert_to_onnx.py`:

*   `MODEL_ID`: The ID of your fine-tuned model on the Hugging Face Hub.
*   `OUTPUT_DIR`: The directory where the ONNX model will be saved.
*   `OPSET`: The ONNX opset version.

### Execution

To run the conversion, simply execute the script:

```bash
python convert_to_onnx.py
```

## Deployment

The `deploy.py` script deploys the ONNX model to a managed online endpoint in Azure Machine Learning.

### User Settings

Before running the script, you need to update the following variables in `deploy.py`:

*   `SUBSCRIPTION_ID`: Your Azure subscription ID.
*   `RESOURCE_GROUP`: The name of your resource group.
*   `WORKSPACE_NAME`: The name of your Azure ML workspace.
*   `COMPUTE_NAME`: The name of your compute cluster.
*   `ENDPOINT_NAME`: The name of the online endpoint.
*   `DEPLOYMENT_NAME`: The name of the deployment.
*   `INSTANCE_TYPE`: The VM size for the deployment.
*   `JOB_RUN_NAME`: The name of the conversion job run.

### Execution

The deployment script performs the following steps:

1.  **Submits a conversion job**: Runs `convert_to_onnx.py` as an Azure ML command job.
2.  **Registers the ONNX model**: Registers the output of the conversion job as a model in your workspace.
3.  **Defines the environment**: Creates an environment from the `conda.yaml` file.
4.  **Creates an endpoint**: Creates a new managed online endpoint.
5.  **Deploys the model**: Deploys the registered model to the endpoint.

To run the deployment, execute the script:

```bash
python deploy.py
```

## Inference



*   **`init()`**: This function is called when the service starts. It loads the model and sets it to evaluation mode.
*   **`run(raw_data)`**: This function is called for each incoming request. It decodes the base64-encoded tensor from the JSON payload, runs the model, and returns the embeddings.

### `Test.py`

The `Test.py` script provides a simple example of how to send a request to the scoring URI.

#### User Settings

Before running the script, you need to update the following variables in `Test.py`:

*   `scoring_uri`: The scoring URI of your deployed endpoint.
*   `key`: The primary key for your endpoint.

#### Execution

The script creates a dummy tensor, encodes it as a base64 string, and sends it to the endpoint. To run the test, execute the script:

```bash
python Test.py
```

This will print the response from the endpoint, which contains the model's embeddings.
