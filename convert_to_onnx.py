# convert_to_onnx.py
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.onnx import export
import mlflow
import mlflow.onnx
from mlflow.models.signature import infer_signature
# from azure.ai.ml import MLClient
# from azure.identity import DefaultAzureCredential

# ------------------------------------------------------------------
# USER SETTINGS – edit these
# ------------------------------------------------------------------
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"   # <-- your HF repo
OUTPUT_DIR = Path("onnx_model")
OPSET = 14
# ------------------------------------------------------------------

def main():
    # Start MLflow run for tracking
    with mlflow.start_run() as run:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

        # Log parameters
        mlflow.log_param("model_id", MODEL_ID)
        mlflow.log_param("opset", OPSET)

        # Export to ONNX
        onnx_path = OUTPUT_DIR / "model.onnx"
        export(
            preprocessor=tokenizer,
            model=model,
            config="default",          # works for most HF models
            opset=OPSET,
            output=onnx_path
        )

        print(f"ONNX model saved to {onnx_path}")

        # Create a dummy input for signature inference
        dummy_input = tokenizer("This is a test sentence.", return_tensors="pt")
        dummy_output = model(**dummy_input)

        # Infer model signature
        signature = infer_signature(dummy_input, dummy_output)

        # Log the ONNX model using MLflow
        mlflow.onnx.log_model(
            onnx_model=str(onnx_path),
            artifact_path="onnx_model",
            signature=signature,
            registered_model_name="terramind-onnx-model" # Register the model for easy deployment
        )

        print(f"MLflow Model logged and registered as 'terramind-onnx-model' in run {run.info.run_id}")

if __name__ == "__main__":
    main()