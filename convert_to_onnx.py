# convert_to_onnx.py
import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.onnx import export, FeaturesManager
import mlflow
import mlflow.onnx
from mlflow.models.signature import infer_signature
import torch

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MODEL_ID = os.getenv("HF_MODEL_ID", "ibm-esa-geospatial/TerraMind-1.0-base")
OUTPUT_DIR = Path("onnx_model")
OPSET = int(os.getenv("ONNX_OPSET", "14"))
MODEL_NAME = os.getenv("MODEL_NAME", "terramind-onnx-model")

print("=" * 70)
print("ONNX Conversion Configuration")
print("=" * 70)
print(f"Model ID: {MODEL_ID}")
print(f"Output Directory: {OUTPUT_DIR}")
print(f"ONNX Opset: {OPSET}")
print(f"Registered Model Name: {MODEL_NAME}")
print("=" * 70)

# ------------------------------------------------------------------
# Configure MLflow for Azure ML
# ------------------------------------------------------------------
# When running in Azure ML, the tracking URI is automatically configured
# But we can explicitly set it to ensure proper integration
if "AZUREML_RUN_ID" in os.environ:
    print("\n✓ Running in Azure ML environment")
    print(f"  Run ID: {os.environ['AZUREML_RUN_ID']}")
    # Azure ML automatically sets the tracking URI
else:
    print("\n⚠ Not running in Azure ML environment")
    print("  MLflow will use local tracking")


def main():
    """Convert HuggingFace model to ONNX format and register with MLflow."""
    
    try:
        # Start MLflow run for tracking
        with mlflow.start_run() as run:
            print(f"\n✓ MLflow run started: {run.info.run_id}")
            
            # Create output directory
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            print(f"✓ Output directory created: {OUTPUT_DIR}")

            # ------------------------------------------------------------------
            # Load model and tokenizer
            # ------------------------------------------------------------------
            print("\nLoading HuggingFace model and tokenizer...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
                print(f"✓ Model loaded successfully")
                print(f"  Model type: {type(model).__name__}")
                print(f"  Tokenizer type: {type(tokenizer).__name__}")
            except Exception as e:
                print(f"ERROR: Failed to load model: {e}")
                sys.exit(1)

            # Log parameters
            mlflow.log_param("model_id", MODEL_ID)
            mlflow.log_param("opset", OPSET)
            mlflow.log_param("model_type", type(model).__name__)
            mlflow.log_param("tokenizer_type", type(tokenizer).__name__)

            # ------------------------------------------------------------------
            # Export to ONNX
            # ------------------------------------------------------------------
            print("\nExporting model to ONNX format...")
            onnx_path = OUTPUT_DIR / "model.onnx"
            
            try:
                # Set model to evaluation mode
                model.eval()
                
                # Get proper ONNX configuration
                _, onnx_config_cls = FeaturesManager.check_supported_model_or_raise(
                    model,
                    feature="sequence-classification",
                )
                onnx_config = onnx_config_cls(model.config)
                
                # Export to ONNX
                export(
                    preprocessor=tokenizer,
                    model=model,
                    config=onnx_config,
                    opset=OPSET,
                    output=onnx_path
                )
                print(f"✓ ONNX model exported to: {onnx_path}")
                
                # Verify ONNX file exists and log size
                if onnx_path.exists():
                    file_size_mb = onnx_path.stat().st_size / (1024 * 1024)
                    print(f"  File size: {file_size_mb:.2f} MB")
                    mlflow.log_metric("model_size_mb", file_size_mb)
                else:
                    raise FileNotFoundError(f"ONNX file not created at {onnx_path}")
                    
            except Exception as e:
                print(f"ERROR: Failed to export to ONNX: {e}")
                sys.exit(1)

            # ------------------------------------------------------------------
            # Create model signature
            # ------------------------------------------------------------------
            print("\nCreating model signature...")
            try:
                # Create dummy input for signature inference
                dummy_text = "This is a test sentence for signature inference."
                dummy_input = tokenizer(dummy_text, return_tensors="pt")
                
                # Get model output
                with torch.no_grad():
                    dummy_output = model(**dummy_input)
                
                # Convert output to dictionary format for signature
                # Use detach().cpu() for safe tensor conversion
                output_dict = {
                    "logits": dummy_output.logits.detach().cpu().numpy()
                }
                
                # Infer signature
                signature = infer_signature(
                    model_input={"text": dummy_text},
                    model_output=output_dict
                )
                print(f"✓ Model signature created")
                
            except Exception as e:
                print(f"WARNING: Failed to create signature: {e}")
                print("  Continuing without signature...")
                signature = None

            # ------------------------------------------------------------------
            # Log ONNX model to MLflow
            # ------------------------------------------------------------------
            print("\nLogging ONNX model to MLflow...")
            try:
                mlflow.onnx.log_model(
                    onnx_model=str(onnx_path),
                    artifact_path="model",
                    signature=signature,
                    registered_model_name=MODEL_NAME
                )
                print(f"✓ MLflow model logged successfully")
                print(f"  Registered as: {MODEL_NAME}")
                print(f"  Run ID: {run.info.run_id}")
                
            except Exception as e:
                print(f"ERROR: Failed to log model to MLflow: {e}")
                sys.exit(1)

            # ------------------------------------------------------------------
            # Log additional artifacts
            # ------------------------------------------------------------------
            print("\nLogging additional artifacts...")
            try:
                # Save tokenizer config
                tokenizer.save_pretrained(OUTPUT_DIR / "tokenizer")
                mlflow.log_artifacts(str(OUTPUT_DIR / "tokenizer"), artifact_path="tokenizer")
                print(f"✓ Tokenizer artifacts logged")
                
            except Exception as e:
                print(f"WARNING: Failed to log tokenizer artifacts: {e}")

            print("\n" + "=" * 70)
            print("ONNX Conversion Completed Successfully!")
            print("=" * 70)
            print(f"Model Name: {MODEL_NAME}")
            print(f"Run ID: {run.info.run_id}")
            print(f"ONNX Path: {onnx_path}")
            print("=" * 70)
            
    except Exception as e:
        print(f"\nERROR: Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
