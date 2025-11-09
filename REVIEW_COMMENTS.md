# PR Review Comments Summary

## Critical Issues (Must Fix)

### 1. ONNX Export Config Issue (convert_to_onnx.py)
**Severity:** 🔴 Critical  
**Location:** Lines 86-92  
**Problem:** `transformers.onnx.export` expects an `OnnxConfig` instance, but the code passes the string `"default"`. This will cause `AttributeError: 'str' object has no attribute 'inputs'`.

**Fix Required:**
```python
from transformers.onnx import export, FeaturesManager

# Get the proper ONNX config
_, onnx_config_cls = FeaturesManager.check_supported_model_or_raise(
    MODEL_ID,
    feature="sequence-classification",
)
onnx_config = onnx_config_cls(model.config)

# Export with proper config
export(
    preprocessor=tokenizer,
    model=model,
    config=onnx_config,
    opset=OPSET,
    output=onnx_path,
)
```

### 2. Model Version Retrieval Issue (deploy.py)
**Severity:** High  
**Location:** Line 59  
**Problem:** `ml_client.models.get(name=MODEL_NAME, label="latest")` is likely to fail unless you explicitly set a label. Using `version="latest"` is more robust.

**Fix Required:**
```python
registered_model = ml_client.models.get(name=MODEL_NAME, version="latest")
```

### 3. Model/Payload Incompatibility (Test.py)
**Severity:** High  
**Location:** Lines 47-63  
**Problem:** The deployed MLflow ONNX model expects sequence classification input (text), but the test client sends a base64-encoded geospatial tensor. This mismatch will break inference requests.

**Fix Required:** Either:
- Update test payload to match sequence classification model (text input)
- OR add a custom scoring script for geospatial tensor input
- OR clarify model type and update accordingly

## Minor Issues (Should Fix)

### 4. Bash Read Command Safety (deploy.bash)
**Severity:** 🔵 Trivial  
**Location:** Lines 71, 77, 83, 89  
**Problem:** Missing `-r` flag in `read` commands, which could cause backslash mangling.

**Fix Required:**
```bash
read -r -p "Azure Subscription ID: " AZURE_SUBSCRIPTION_ID
```

### 5. Markdown Formatting (ISSUES_FOUND.md)
**Severity:** 🔵 Trivial  
**Problem:** Missing blank lines around headings and code blocks.

**Fix Required:** Add blank lines before/after headings and code blocks.

### 6. Code Block Language Identifier (README.md)
**Severity:** 🔵 Trivial  
**Location:** Lines 153-164  
**Problem:** Project structure code block missing language identifier.

**Fix Required:**
```text
qcx-terra/
├── deploy.py
...
```

### 7. Environment Existence Check (deploy.py)
**Severity:** Low  
**Location:** Lines checking for environment version="1"  
**Problem:** Brittle check; prefer idempotent `create_or_update`.

**Fix Required:** Use `create_or_update` and rely on `@latest` for jobs.

### 8. Security: Endpoint Key Printing (deploy.py)
**Severity:** Low  
**Problem:** Full endpoint key printed to stdout.

**Fix Required:** Mask or truncate the key in console output.

### 9. Tensor Conversion Safety (convert_to_onnx.py)
**Severity:** Low  
**Problem:** Direct tensor to numpy conversion without detaching.

**Fix Required:**
```python
output_dict = {
    "logits": dummy_output.logits.detach().cpu().numpy()
}
```

## Compliance Issues

### Security Compliance
- 🔴 Generic: Secure Error Handling
- 🔴 Generic: Secure Logging Practices  
- 🔴 Generic: Security-First Input Validation and Data Handling

## Priority Order

1. **Fix ONNX export config** (Critical - will cause runtime failure)
2. **Fix model version retrieval** (High - deployment may fail)
3. **Fix model/payload incompatibility** (High - inference will fail)
4. **Add `-r` flag to read commands** (Medium - robustness)
5. **Fix environment check** (Medium - reliability)
6. **Mask endpoint key in output** (Medium - security)
7. **Fix tensor conversion** (Low - best practice)
8. **Fix markdown formatting** (Low - documentation quality)
9. **Add code block language** (Low - documentation quality)
