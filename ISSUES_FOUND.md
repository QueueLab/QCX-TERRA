# Azure Connectivity Issues Identified in QCX-TERRA

## Issue 1: Subscription ID Not Properly Retrieved

**Location:** `deploy.py` line 17

**Problem:**

```python
SUBSCRIPTION_ID = os.getenv("SUBSCRIPTION_ID")
```

- The code retrieves the subscription ID from environment variable but doesn't validate if it exists
- If the environment variable is not set, `SUBSCRIPTION_ID` will be `None`
- This causes `MLClient` initialization to fail silently or with cryptic errors

**Impact:** MLClient cannot connect to Azure without a valid subscription ID

## Issue 2: Hardcoded Placeholder Values

**Location:** `deploy.py` lines 18-20

**Problem:**

```python
RESOURCE_GROUP   = "<your-resource-group>"
WORKSPACE_NAME   = "<your-aml-workspace>"
COMPUTE_NAME     = "<your-compute-cluster>"
```

- These are placeholder strings that will cause Azure API calls to fail
- No environment variable fallback or validation

**Impact:** All Azure ML operations will fail with resource not found errors

## Issue 3: Missing MLflow Tracking URI Configuration

**Location:** `convert_to_onnx.py`

**Problem:**

- MLflow is used but no tracking URI is configured for Azure ML
- When running in Azure ML compute, MLflow needs to be configured to use Azure ML's tracking system
- Missing `mlflow.set_tracking_uri()` call

**Impact:** MLflow model registration may fail or register to wrong location

## Issue 4: Missing Azure ML Environment Configuration

**Location:** `deploy.py` lines 37-38

**Problem:**

```python
environment="hf-onnx-env@latest",
```

- References an environment that doesn't exist yet
- No code to create this environment first
- The environment definition is commented out (lines 67-71)

**Impact:** Job submission will fail due to missing environment

## Issue 5: Missing Transformers Dependency

**Location:** `requirements.txt`

**Problem:**

- `convert_to_onnx.py` imports `transformers` library but it's not in requirements.txt
- Missing `mlflow` package as well

**Impact:** Conversion job will fail with ImportError

## Issue 6: Missing Error Handling

**Location:** Throughout `deploy.py`

**Problem:**

- No try-except blocks around Azure API calls
- No validation of job completion before proceeding
- No checks for model registration success

**Impact:** Failures are not gracefully handled, making debugging difficult

## Issue 7: Test Script Issues

**Location:** `Test.py`

**Problem:**

- Missing `import numpy as np`
- Hardcoded placeholder values for endpoint URI and key
- No error handling

**Impact:** Test script cannot run successfully
