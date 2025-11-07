# convert_to_onnx.py
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.onnx import export
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

# ------------------------------------------------------------------
# USER SETTINGS – edit these
# ------------------------------------------------------------------
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"   # <-- your HF repo
OUTPUT_DIR = Path("onnx_model")
OPSET = 14
# ------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    # Export to ONNX
    export(
        preprocessor=tokenizer,
        model=model,
        config="default",          # works for most HF models
        opset=OPSET,
        output=OUTPUT_DIR / "model.onnx"
    )

    print(f"ONNX model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()