# score.py
import json
import numpy as np
import torch
from terratorch import BACKBONE_REGISTRY
import base64  # For tensor serialization

# ------------------------------------------------------------------
# USER SETTINGS – match your fine-tune
# ------------------------------------------------------------------
MODEL_NAME = "terramind_v1_base_tim"
MODALITIES = ['S2L2A']  # Expected inputs
TIM_MODALITIES = ['LULC']
CHECKPOINT_PATH = "./model.ckpt"  # Mounted from registered model
BATCH_SIZE = 1
IMG_SIZE = 224
# ------------------------------------------------------------------

def init():
    global model, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BACKBONE_REGISTRY.build(
        MODEL_NAME,
        pretrained=False,  # Load from checkpoint
        modalities=MODALITIES,
        tim_modalities=TIM_MODALITIES
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device)["state_dict"])
    model.to(device)
    model.eval()

def run(raw_data: str):
    data = json.loads(raw_data)
    inputs = {}
    for mod in MODALITIES:
        # Decode base64 tensor (sent as JSON: {"S2L2A": "base64_string"})
        tensor_b64 = data.get(mod, "")
        tensor_bytes = base64.b64decode(tensor_b64)
        tensor_np = np.frombuffer(tensor_bytes, dtype=np.float32).reshape((BATCH_SIZE, -1, IMG_SIZE, IMG_SIZE))
        inputs[mod] = torch.from_numpy(tensor_np).to(device)
    
    with torch.no_grad():
        outputs = model(inputs)
    
    # Return embeddings as list (or adapt for generations)
    embeddings = outputs["embeddings"].cpu().numpy().tolist() if "embeddings" in outputs else outputs.tolist()
    return {"embeddings": embeddings, "shape": list(outputs.shape) if hasattr(outputs, 'shape') else "dict"}