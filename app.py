# app.py
import sys
from fastapi import FastAPI, HTTPException
from transformers import pipeline
from fastapi.testclient import TestClient
import uvicorn

# Create a FastAPI app
app = FastAPI(title="Hugging Face Sentiment Analysis API")

# Load the Hugging Face sentiment analysis model
model_name = "vishodi/First-Name-Classification"
classifier = pipeline("sentiment-analysis", model=model_name)

# Define the /predict endpoint
@app.get("/predict")
def predict(text: str):
    try:
        result = classifier(text)
        # Return the first result from the pipeline
        return result[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Inline test code for local testing
def run_tests():
    client = TestClient(app)
    # Example test query
    response = client.get("/predict", params={"text": "I love AI!"})
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    print("Test response:", data)

if __name__ == "__main__":
    # Run tests if the 'test' argument is provided
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        # Run the API server
        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
