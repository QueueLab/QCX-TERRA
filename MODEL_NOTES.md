# Model Type and Usage Notes

## Current Model Configuration

The deployment scripts are configured to use the **TerraMind-1.0-base** model from HuggingFace, which is a sequence classification model designed for text input.

### Model Details

- **Model ID:** `ibm-esa-geospatial/TerraMind-1.0-base`
- **Model Type:** Sequence Classification
- **Input Format:** Text strings
- **Output Format:** Classification logits

## Important Considerations

### 1. Model vs. Test Payload Mismatch

The original test script (`Test.py`) was sending base64-encoded geospatial tensor data (Sentinel-2 satellite imagery), but the deployed model expects text input for sequence classification. This has been corrected in the updated version.

### 2. For Geospatial Inference

If you need to perform geospatial inference with satellite imagery:

**Option A: Use a Different Model**
- Deploy a model specifically trained for geospatial/remote sensing tasks
- Update the `HF_MODEL_ID` environment variable to point to the geospatial model
- Ensure the model architecture supports image/tensor input

**Option B: Create Custom Scoring Script**
- Keep the current model but add a custom scoring script
- The scoring script would handle preprocessing of geospatial data
- This requires modifying the deployment to include a custom inference script

**Option C: Fine-tune TerraMind for Your Use Case**
- Use the `terramind_config.yaml` configuration
- Fine-tune the model on your specific geospatial task
- Deploy the fine-tuned model with appropriate input handling

### 3. Current Test Script

The updated `Test.py` now correctly sends text input that matches the sequence classification model:

```python
test_samples = [
    "This is a test sentence for model inference.",
    "Another example text for classification.",
    "The model will classify these text inputs."
]
```

### 4. Recommended Next Steps

1. **Clarify your use case:**
   - Are you doing text classification? → Current setup is correct
   - Are you doing geospatial analysis? → Need to modify model or add custom scoring

2. **For geospatial use:**
   - Review the TerraMind model documentation
   - Check if there's a vision-based variant
   - Consider using a model like `microsoft/resnet-50` or similar for image tasks

3. **Update configuration:**
   - Modify `HF_MODEL_ID` in `.env` or environment variables
   - Update `convert_to_onnx.py` to use appropriate model classes (e.g., `AutoModelForImageClassification`)
   - Update `Test.py` payload format to match your model's expected input

## Example: Switching to Image Classification

If you want to deploy an image classification model instead:

```bash
# Set environment variable
export HF_MODEL_ID="microsoft/resnet-50"

# Update convert_to_onnx.py imports
# Change: from transformers import AutoModelForSequenceClassification
# To: from transformers import AutoModelForImageClassification

# Update Test.py to send image data
# Format: base64-encoded image or PIL Image
```

## References

- [TerraMind Model Card](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [Azure ML Custom Scoring Scripts](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-online-endpoints)
