#!/usr/bin/env python3
"""
Test script for TerraMind ONNX endpoint deployed on Azure ML.

Note: This model is a sequence classification model that expects text input,
not geospatial tensor data. For geospatial inference, a custom scoring script
would be needed.
"""
import os
import sys
import json
import requests


def load_endpoint_config():
    """Load endpoint configuration from file or environment variables."""
    config = {}
    
    # Try to load from endpoint_details.txt first
    if os.path.exists("endpoint_details.txt"):
        print("Loading endpoint configuration from endpoint_details.txt...")
        with open("endpoint_details.txt", "r") as f:
            for line in f:
                if "Scoring URI:" in line:
                    config["scoring_uri"] = line.split("Scoring URI:")[1].strip()
                elif "Primary Key:" in line:
                    config["key"] = line.split("Primary Key:")[1].strip()
    
    # Override with environment variables if set
    if os.getenv("SCORING_URI"):
        config["scoring_uri"] = os.getenv("SCORING_URI")
    if os.getenv("ENDPOINT_KEY"):
        config["key"] = os.getenv("ENDPOINT_KEY")
    
    # Validate configuration
    if not config.get("scoring_uri") or config["scoring_uri"].startswith("<"):
        print("ERROR: Scoring URI not configured")
        print("Please set SCORING_URI environment variable or update endpoint_details.txt")
        sys.exit(1)
    
    if not config.get("key") or config["key"].startswith("<"):
        print("ERROR: Endpoint key not configured")
        print("Please set ENDPOINT_KEY environment variable or update endpoint_details.txt")
        sys.exit(1)
    
    return config


def create_test_payload():
    """Create a test payload for sequence classification model."""
    # The model expects text input for sequence classification
    # This is compatible with the HuggingFace model deployed
    test_samples = [
        "This is a test sentence for model inference.",
        "Another example text for classification.",
        "The model will classify these text inputs."
    ]
    
    # Create payload in the format expected by MLflow ONNX models
    # For sequence classification, we send text data
    payload = {
        "inputs": test_samples
    }
    
    return json.dumps(payload)


def test_endpoint(scoring_uri: str, key: str):
    """Test the Azure ML endpoint with a sample request."""
    print("\n" + "=" * 70)
    print("Testing Azure ML Endpoint")
    print("=" * 70)
    print(f"Endpoint URI: {scoring_uri}")
    print("=" * 70)
    
    # Create test payload
    print("\nCreating test payload...")
    payload = create_test_payload()
    print(f"✓ Payload created")
    print(f"  Sample input: {json.loads(payload)['inputs'][0]}")
    
    # Set up headers
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    # Send request
    print("\nSending request to endpoint...")
    try:
        response = requests.post(scoring_uri, data=payload, headers=headers, timeout=30)
        
        print(f"\nResponse Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✓ Request successful!")
            print("\nResponse:")
            try:
                result = response.json()
                print(json.dumps(result, indent=2))
            except json.JSONDecodeError:
                print(response.text)
        else:
            print(f"✗ Request failed!")
            print(f"\nError Response:")
            print(response.text)
            sys.exit(1)
            
    except requests.exceptions.Timeout:
        print("✗ Request timed out after 30 seconds")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"✗ Request failed: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
    print("\nNote: This model performs sequence classification on text input.")
    print("For geospatial tensor inference, you would need to:")
    print("  1. Deploy a model trained for geospatial data")
    print("  2. Create a custom scoring script")
    print("  3. Update the payload format accordingly")


def main():
    """Main function."""
    print("TerraMind ONNX Endpoint Test Script")
    print("\nNote: Testing sequence classification model with text input")
    
    # Load configuration
    config = load_endpoint_config()
    
    # Test endpoint
    test_endpoint(config["scoring_uri"], config["key"])


if __name__ == "__main__":
    main()
