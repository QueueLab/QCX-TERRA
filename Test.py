import requests, json, base64
# Get URI/key from ml_client
scoring_uri = "<your-endpoint-uri>/score"
key = "<your-primary-key>"

# Dummy tensor (replace with real raster load + base64)
dummy_tensor = np.random.rand(1, 6, 224, 224).astype(np.float32)  # 6 bands for S2L2A
tensor_b64 = base64.b64encode(dummy_tensor.tobytes()).decode()
payload = json.dumps({"S2L2A": tensor_b64})

headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
resp = requests.post(scoring_uri, data=payload, headers=headers)
print(resp.json())